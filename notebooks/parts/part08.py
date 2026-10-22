# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part VIII (first part): the final TDI model. The candidate grid (8.1), the stack (8.2), the reproducibility bug in the stack (8.3)."""

from notebooks.notebook_builder import code, md

SETUP8 = '''
# The objects every section of this part reads, for CYP3A4: the nested plan of 6.3 and its cache of predictions, the similarity matcher and the confidence z of 6.4, and the positive rate of the scored set (2.10). The names carry a
# 8 so that they do not overwrite the ones of Part VI.
import numpy as np
import pandas as pd

from cypmode.challenge.label_confidence import confidence_z, confident_weights
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.nested import OUTER_REFIT, BestSingle, NestedPlan, PredictionCache, StackedLogit, inner_oof, nested_evaluate
from cypmode.challenge.nested_runners import FEATURE_SETS, LGBM_PRESETS, default_candidates
from cypmode.challenge.validation_matching import SimilarityMatcher

NESTED_ROOT8 = ROOT / "data/tdi_track/outputs/nested/CYP3A4"
plan8 = NestedPlan.load(NESTED_ROOT8 / "plan.npz")
cache8 = PredictionCache(NESTED_ROOT8 / "cache", plan8)
train8 = pd.read_csv(ROOT / "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
store8 = np.load(ROOT / "data/tdi_track/outputs/caches/feature_store.npz", allow_pickle=True)
matcher8 = SimilarityMatcher(store8["morgan_bits"].astype(np.float32), plan8.external)
prevalence8 = float(estimate_from_csv(ROOT / "data/tdi_track/inputs/leaderboard_snapshots/interim_CYP3A4_2026-10-04.csv")["prevalence"])
z8 = np.concatenate([confidence_z(train8, "CYP3A4"), np.full(len(plan8.external), np.nan)])
def weights8(z0):
    return confident_weights(matcher8, z8, z0)
candidates8 = default_candidates()
names8 = [c.name for c in candidates8]
tasks_per_candidate = len(plan8.tasks())
plan_d8 = NestedPlan.load(ROOT / "data/tdi_track/outputs/nested/CYP2D6/plan.npz")
cache_d8 = PredictionCache(ROOT / "data/tdi_track/outputs/nested/CYP2D6/cache", plan_d8)
setups8 = {"CYP3A4": (plan8, cache8, matcher8, prevalence8, z8),
           "CYP2D6": (plan_d8, cache_d8, SimilarityMatcher(store8["morgan_bits"].astype(np.float32), plan_d8.external), float(estimate_from_csv(ROOT / "data/tdi_track/inputs/leaderboard_snapshots/interim_CYP2D6_2026-10-04.csv")["prevalence"]),
                      np.concatenate([confidence_z(train8, "CYP2D6"), np.full(len(plan_d8.external), np.nan)]))}
print(f"{len(names8)} default members; the plan has {tasks_per_candidate} tasks per member ({len(plan8.outer_folds())} outer folds of the main design, 5 repeats of the refit)")
'''

GRID_TABLE = '''
# The 23 default members, read from the project's grid (cypmode/challenge/nested_runners.py, default_candidates). For each: its family, the feature blocks it reads (and how many columns that is in the feature store), its
# settings, and its score under the yardstick of 6.4: the mean outer AUC over the 15 outer folds, a member scored alone ("best single" selector), at the two confidence cut-offs.
def columns_of(blocks):
    return int(sum(store8[b].shape[1] for b in blocks))

set_of_blocks = {tuple(blocks): name for name, blocks in FEATURE_SETS.items()}
member_rows = []
for candidate in candidates8:
    scores = {}
    for z0 in (1.0, 1.96):
        table = nested_evaluate(cache8, [candidate.name], [BestSingle()], weights8(z0), prevalence8, "auc")
        scores[z0] = float(table["outer_score"].mean())
    member_rows.append({"member": candidate.name, "family": candidate.family.replace("_", " "), "feature set": set_of_blocks.get(tuple(candidate.blocks), "-") if candidate.blocks else "fingerprint similarity",
                        "columns": columns_of(candidate.blocks) if candidate.blocks else 0, "settings": ", ".join(f"{k}={v}" for k, v in candidate.params.items() if k in ("k", "C", "n_estimators", "num_leaves", "learning_rate")) or "-",
                        "outer AUC at z0 1.0": scores[1.0], "outer AUC at z0 1.96": scores[1.96]})
member_table = pd.DataFrame(member_rows)
show_table(member_table, floatfmt=".3f")
family_summary = member_table.groupby("family")[["outer AUC at z0 1.0", "outer AUC at z0 1.96"]].agg(["min", "max", "count"]).round(3)
n_members, n_families = len(member_table), member_table["family"].nunique()
best_member = member_table.sort_values("outer AUC at z0 1.0", ascending=False).iloc[0]
worst_member = member_table.sort_values("outer AUC at z0 1.0").iloc[0]
set_widths = {name: columns_of(blocks) for name, blocks in FEATURE_SETS.items()}
'''

STACK_TOY = '''
# Why the blend is penalised. A toy in which the truth is known: 600 compounds, 23 members of which 3 carry the signal (the same signal seen through three noisy windows) and 20 are pure noise, as the weaker
# members of a real grid partly are. A logistic blend is fitted on the first 400 compounds with an L2 penalty (all weights shrunk, none to zero) and with an L1 penalty (many weights exactly zero), and scored on the other 200,
# for 200 random draws of the data. The penalty strength is the best of the project's grid on a five-fold cross-validation of the training part, as in the stack.
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import sklearn
from sklearn.linear_model import LogisticRegression

def liblinear_logit(penalty, c, tol, seed):
    """A liblinear logistic regression; scikit-learn 1.8 replaced the argument `penalty` by `l1_ratio` (0 is L2, 1 is L1)."""
    settings = {"C": c, "solver": "liblinear", "max_iter": 20000, "tol": tol, "random_state": seed}
    if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8):
        return LogisticRegression(l1_ratio=1.0 if penalty == "l1" else 0.0, **settings)
    return LogisticRegression(penalty=penalty, **settings)


def blend_scores(seed, n_noise=20):
    rng = np.random.default_rng(seed)
    n = 600
    y = rng.integers(0, 2, n)
    informative = (y[:, None] - 0.5) * 0.8 + rng.normal(size=(n, 3))
    X = np.hstack([informative, rng.normal(size=(n, n_noise))])
    train, test = np.arange(400), np.arange(400, 600)
    out = {}
    for penalty in ("l1", "l2"):
        best = None
        for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            model = liblinear_logit(penalty, c, 1e-9, 0)
            value = cross_val_score(model, X[train], y[train], scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
            if best is None or value > best[0] + 1e-9:
                best = (value, c)
        model = liblinear_logit(penalty, best[1], 1e-9, 0).fit(X[train], y[train])
        out[penalty] = {"auc": roc_auc_score(y[test], model.decision_function(X[test])), "nonzero noise weights": int((np.abs(model.coef_[0][3:]) > 1e-9).sum()), "weight on the signal": float(np.abs(model.coef_[0][:3]).sum())}
    chosen = int(np.argmax([roc_auc_score(y[train], X[train][:, j]) for j in range(X.shape[1])]))
    out["best single member"] = {"auc": roc_auc_score(y[test], X[test][:, chosen]), "nonzero noise weights": int(chosen >= 3), "weight on the signal": float(chosen < 3)}
    return out
toy_runs = [blend_scores(seed) for seed in range(60)]
toy_table = pd.DataFrame([{"rule": {"l1": "L1-penalised blend", "l2": "L2-penalised blend", "best single member": "best single member"}[p], "mean test AUC": float(np.mean([r[p]["auc"] for r in toy_runs])),
                           "mean number of noise members with a non-zero weight (of 20)": float(np.mean([r[p]["nonzero noise weights"] for r in toy_runs]))} for p in ("l1", "l2", "best single member")])
show_table(toy_table, floatfmt=".3f")
toy_values = toy_table.set_index("rule")
'''

STACK_REAL = '''
# The stack on real compounds, for the first outer fold of the first repeat. The inner out-of-fold scores of the 23 members on the outer-training compounds are put on a common scale (each member's empirical distribution
# function, so that a probability, a margin and a regression value can be blended), turned into log-odds, and blended by a logistic regression whose penalty (L1 or L2) and strength C are chosen by a five-fold
# cross-validation of the blend itself (cypmode/challenge/nested.py, StackedLogit). The chosen setting, the weights, and the blend's score on the outer-validation compounds, against the best single member and the average.
from scipy.stats import rankdata

repeat0, outer0 = plan8.outer_folds()[0]
rows_in, oof = inner_oof(cache8, names8, repeat0, outer0)
refit = plan8.task(repeat0, outer0, OUTER_REFIT)
y_in, y_out = plan8.labels(rows_in), plan8.labels(refit.evaluate)
outer_matrix = np.column_stack([cache8.get(n, refit) for n in names8])
w_in = None                                                                  # the plain AUC here, to keep the example about the blend and not about the yardstick
selection = StackedLogit().fit(names8, oof, y_in, w_in, "auc")
blend_out = selection.predict(outer_matrix)
single_out = [roc_auc_score(y_out, outer_matrix[:, j]) for j in range(len(names8))]
single_in = [roc_auc_score(y_in, oof[:, j]) for j in range(len(names8))]
mean_rank = np.mean([rankdata(outer_matrix[:, j]) for j in range(len(names8))], axis=0)
stack_rows = [{"quantity": "inner AUC of the blend (cross-validated inside the training part)", "value": selection.inner_score}, {"quantity": "outer AUC of the blend", "value": roc_auc_score(y_out, blend_out)},
              {"quantity": "outer AUC of the best single member (chosen on the inner scores)", "value": single_out[int(np.argmax(single_in))]},
              {"quantity": "outer AUC of the average of the ranks", "value": roc_auc_score(y_out, mean_rank)}]
show_table(pd.DataFrame(stack_rows), floatfmt=".4f")
weights_table = pd.DataFrame([{"member": k, "weight on the log-odds scale": v} for k, v in selection.chosen.items()])
show_table(weights_table, floatfmt=".3f")
stack_values = {row["quantity"]: row["value"] for row in stack_rows}
'''

LIBLINEAR_BUG = '''
# The reproducibility bug of the stack, reproduced. The solver the stack uses for its L1 and L2 penalties (liblinear) visits the coordinates of the problem in a random order, and it stops when the change falls below a
# tolerance. With the library's default tolerance, 1e-4, the point where it stops depends on that random order; the problem is convex, so with a tight tolerance every order ends at the same optimum. The cell fits the same L1 blend
# (the setting the stack chose for this fold) with ten different random orders at the default tolerance and at 1e-9, on the inner scores of the first fold of CYP2D6, and measures how much the blend and its AUC move.
from sklearn.metrics import matthews_corrcoef

from cypmode.challenge.nested import _EmpiricalCdf, call_top_fraction

plan_2d6, cache_2d6 = plan_d8, cache_d8
rep_2d6, out_2d6 = plan_2d6.outer_folds()[0]
rows_2d6, oof_2d6 = inner_oof(cache_2d6, names8, rep_2d6, out_2d6)
refit_2d6 = plan_2d6.task(rep_2d6, out_2d6, OUTER_REFIT)
y_in_2d6, y_out_2d6 = plan_2d6.labels(rows_2d6), plan_2d6.labels(refit_2d6.evaluate)
outer_2d6 = np.column_stack([cache_2d6.get(n, refit_2d6) for n in names8])
cdfs_2d6 = [_EmpiricalCdf(oof_2d6[:, j]) for j in range(len(names8))]
logit = StackedLogit._logit
design = lambda m: np.column_stack([logit(cdfs_2d6[j](m[:, j])) for j in range(m.shape[1])])
z_in, z_out = design(oof_2d6), design(outer_2d6)
bug_rows = []
for tolerance in (1e-4, 1e-9):
    coefficients, aucs, calls_mcc, calls = [], [], [], []
    for seed in range(10):
        model = liblinear_logit("l1", 0.03, tolerance, seed).fit(z_in, y_in_2d6)
        coefficients.append(model.coef_[0].copy())
        scores_out = model.decision_function(z_out)
        aucs.append(roc_auc_score(y_out_2d6, scores_out))
        called = call_top_fraction(scores_out, y_in_2d6.mean()).astype(int)
        calls.append(called)
        calls_mcc.append(matthews_corrcoef(y_out_2d6, called))
    coefficients = np.array(coefficients)
    bug_rows.append({"tolerance": f"{tolerance:g}", "outer AUC, lowest over 10 orders": min(aucs), "outer AUC, highest": max(aucs), "compounds called by some orders and not by others": int((np.array(calls).any(axis=0) & ~np.array(calls).all(axis=0)).sum()), "of the validation compounds": len(y_out_2d6),
                     "largest spread of one weight": float((coefficients.max(axis=0) - coefficients.min(axis=0)).max()), "different blends (to 6 decimals)": len({tuple(np.round(c, 6)) for c in coefficients})})
bug_table = pd.DataFrame(bug_rows)
show_table(bug_table.assign(**{"largest spread of one weight": bug_table["largest spread of one weight"].map(lambda v: f"{v:.1e}")}), floatfmt=".5f")
bug_default, bug_tight = bug_rows[0], bug_rows[1]
'''


CELLS = [
    md("""
# Part VIII. The final TDI model

Part VII ended with a list of ideas that no design of its time could separate from noise, and Part VI built the measurement that can. This part is the model that was built and judged with that measurement. It has seven sections. Section 8.1 describes the grid of candidate models, the 23 members of the *default stack*. Section 8.2 describes the blend that combines their scores and chooses how much each member counts without ever seeing the compounds it is scored on. Section 8.3 tells of a bug in the solver of that blend which made two runs of the same code disagree, and how it was found and removed. Section 8.4 turns the blended scores into yes/no calls, 8.5 turns them into probabilities, 8.6 describes the files that are written for submission and the checks they pass, and 8.7 gives the results with their uncertainty.

What this part hands on is the default stack and the final build around it. Whether anything is added to the default stack is the business of the gates (6.7), applied in Parts IX and X to every proposed addition; the final build (`cypmode/challenge/final_tdi_build.py`) reads their verdicts and applies one rule per enzyme, so that the stack of the submitted files is whatever the verdicts say and not a choice made by hand. Section 8.7 reads the stack the build chose from its manifest, and every number of this part that depends on it is read from the build's files when the notebook is built.
"""),
    md("""
## 8.1 The candidate grid: 23 members and their feature sets

> **Field:** Machine learning · Cheminformatics

*Where this section sits:* 6.3 built the nested plan and its cache of predictions, and 6.2 showed that a model chosen among many on the score that chose it is flattered. This section describes what is in the cache: the candidates, the *members* of the default stack. They are fitted once for every task of the plan, 80 fits per member and enzyme, and every later analysis of this part reads their predictions and fits nothing. The grid is varied on purpose, in three ways.

**Different information.** A feature set is a list of blocks of the feature store (section 3.7), and the members read six of them. `desc` is the 217 RDKit descriptors. `A` adds the Morgan fingerprint in counts (section 3.1). `A+fps` adds the motif block of section 5.1 and the five other fingerprints of section 3.4 (MACCS keys, Avalon, topological torsions, the feature-Morgan counts and the extended reduced graph). `A+ext` adds the five probabilities of the external panel block of 7.7, `A+fps+ext` both, and `A+fps+ext+alerts` also adds the structural alerts of section 1.10. The last cell of this section counts their columns from the store.

**Different learners.** The simplest members use no feature set at all, only the Tanimoto similarity between fingerprints (section 3.2): a nearest-neighbour vote of the 5, 15 or 40 most similar training compounds, and a support vector machine with the Tanimoto similarity as its kernel (section 4.5) at three strengths of regularisation. The tree members are the LightGBM of section 4.3 in two presets, a *deep* one (300 trees, learning rate 0.04, 31 leaves, a random 80 percent of the rows and 30 percent of the columns for each tree) and a *shallow* one (600 trees, learning rate 0.02, 8 leaves, a stronger penalty on the leaf values), and the extremely randomised trees and the random forest of section 4.4. Their settings are fixed in advance; no setting is tuned inside a task, and the blend decides how much each member deserves.

**Different targets.** Most members classify the label. Two read the labels of the 10 nearest neighbours as extra columns (the construction of 7.3). Four are *regressions*, and they use information that the yes/no label throws away. Because the label is a function of two measured numbers (section 2.3), a model can predict those numbers instead: one member predicts the *shift* $t - \\max(d, 4)$ directly, and one predicts the TDI-arm value $t$ and the direct-inhibition value $d$ with two regressors and scores the compound by $t - \\max(d, 4)$, the quantity the label thresholds. A compound that is a long way above the threshold is ranked above one that barely clears it, which a classifier fitted to the yes/no label cannot know. The members are listed from the simplest to the most complex, because one of the selectors of 6.2 breaks ties by that order.

The cell lists the 23 members with their family, feature set, settings and score. The score is the mean outer AUC over the 15 outer folds of the plan when the member is used alone, with the yardstick of 6.4 (similarity to the test compounds, the scored positive rate, confident labels only) at the two cut-offs.
"""),
    code(SETUP8),
    code(GRID_TABLE),
    md("""
The {{=n_members}} members come from {{=n_families}} families. Used alone, under the yardstick, the best is `{{=best_member['member']}}` with a mean outer AUC of {{=best_member['outer AUC at z0 1.0']:.3f}} at $z_0 = 1.0$, and the weakest is `{{=worst_member['member']}}` with {{=worst_member['outer AUC at z0 1.0']:.3f}}. The order is by information, much more than by learner. LightGBM on the descriptors alone scores {{=member_table.set_index('member').loc['lgbm_desc_deep', 'outer AUC at z0 1.0']:.3f}}; with the Morgan counts, {{=member_table.set_index('member').loc['lgbm_A_deep', 'outer AUC at z0 1.0']:.3f}}; with the other fingerprints, {{=member_table.set_index('member').loc['lgbm_A+fps_deep', 'outer AUC at z0 1.0']:.3f}} and with the external block and the alerts as well, {{=member_table.set_index('member').loc['lgbm_A+fps+ext+alerts_deep', 'outer AUC at z0 1.0']:.3f}}. The two presets are nearly interchangeable ({{=member_table.set_index('member').loc['lgbm_A+fps_deep', 'outer AUC at z0 1.0']:.3f}} deep against {{=member_table.set_index('member').loc['lgbm_A+fps_shallow', 'outer AUC at z0 1.0']:.3f}} shallow on `A+fps`). The similarity-only members are the weakest ({{=member_table.set_index('member').loc['knn_tanimoto_k5', 'outer AUC at z0 1.0']:.3f}} for 5 neighbours up to {{=member_table.set_index('member').loc['svm_tanimoto_C1', 'outer AUC at z0 1.0']:.3f}} for the best support vector machine) and the most different from the others, which is what makes them useful in a blend. At the stricter cut-off $z_0 = 1.96$ every score is higher, since the compounds whose label is a coin flip are no longer counted against the model (6.4), and the order is the same. The feature sets have {{=set_widths['desc']}} (`desc`), {{=set_widths['A']:,}} (`A`) and {{=set_widths['A+fps']:,}} (`A+fps`) columns.

Why not take the best member? Because "the best" is a choice made on the scores of the folds, and 6.2 measured what that costs, and because members that are individually weaker can carry information the best does not. The next section is the answer.

*On what actually ran.* The grid is `default_candidates` in `cypmode/challenge/nested_runners.py`, with `FEATURE_SETS`, `LGBM_PRESETS`, `SVM_C`, `KNN_K` and the families `knn_tanimoto`, `svm_tanimoto`, `lgbm_clf`, `extratrees`, `rf`, `lgbm_reg_shift` and `lgbm_reg_arms`, each a branch of `fit_predict`. The cache of predictions is filled by that module (and by the process-isolated runners `nested_runners_torch.py` and `nested_runners_chemprop.py` for the neural members of Part IX), and the plan identifies it: if the plan changes, the final build refuses to run (`final_tdi_build.py` asserts that the plan it rebuilds has the stored identity).
"""),
    md("""
## 8.2 The stack: a logistic blend of the members, with its penalty chosen inside

> **Field:** Statistics · Machine learning

*Where this section sits:* 8.1 produced 23 scores for every compound. One score is needed. The options are to take the best member, to average the members, or to learn a weighted blend. Learning the blend from the members' own out-of-fold predictions is **stacked generalisation**, or stacking (Wolpert, 1992): a second model, the **meta-learner**, is fitted to the predictions of the first-level models. This section describes the meta-learner of the project, the reason for its penalty, and what it does to a score.

**The construction, step by step.** For one outer fold, the cache gives the *inner out-of-fold* scores of the 23 members on the outer-training compounds: each compound scored by models that did not see it (6.3). (1) The members' raw scores are on different scales, a probability, a margin of the support vector machine, a regression value. Each member's scores are mapped through its own empirical distribution function $F_j$, estimated on its inner scores only, so that a score becomes its position in that member's distribution, a number between 0 and 1, and the same map is later applied to the outer scores. (2) The position is turned into log-odds, $z_{ij} = \\log\\!\\big(F_j(s_{ij}) / (1 - F_j(s_{ij}))\\big)$, with $s_{ij}$ the raw score of member $j$ for compound $i$, clipped away from 0 and 1. (3) A logistic regression of the label on the 23 log-odds is fitted. Write $P(y_i = 1)$ for its probability that compound $i$ is a time-dependent inhibitor and $\\sigma$ for the logistic function, which turns any number $t$ into a probability between 0 and 1,

$$P(y_i = 1) \\;=\\; \\sigma\\!\\Big(b + \\sum_{j=1}^{23} w_j\\, z_{ij}\\Big), \\qquad \\sigma(t) = \\frac{1}{1 + e^{-t}},$$

The weights $w_j$ and the intercept $b$ are found by minimising the penalised loss

$$C \\sum_i \\omega_i\\, \\log\\!\\Big(1 + e^{-\\tilde y_i\\,(b + \\sum_j w_j z_{ij})}\\Big) \\;+\\; R(w), \\qquad R(w) = \\sum_j |w_j| \\ \\text{(L1)} \\quad\\text{or}\\quad \\tfrac{1}{2} \\sum_j w_j^2 \\ \\text{(L2)},$$

where $R(w)$ is the penalty on the weights, in one of two forms, $\\tilde y_i = \\pm 1$ is the label written as plus or minus one, $\\omega_i$ is the weight of compound $i$ in the yardstick of 6.4, and $C$ is a constant that sets how much the penalty counts: a small $C$ is a strong penalty and weights near zero, a large $C$ almost none. (4) The penalty type and $C$ are chosen by a five-fold cross-validation *of the blend itself*, on the inner scores: the grid is L1 and L2 with $C$ in {0.003, 0.01, 0.03, 0.1, 0.3, 1}, the criterion is the weighted AUC of the cross-validated blend scores, and a tie keeps the earlier, sparser setting. (5) The threshold of the blend is chosen on the same cross-validated scores (8.4). The outer compounds are never used for any of the five steps.

**Why a penalty, and why L1.** With 23 members and a few thousand compounds, an unpenalised blend fits the noise of the members' errors. The two penalties differ in what they do to the weights. For the simplest case, a regression with orthonormal columns, the lasso (L1, Tibshirani, 1996) shrinks each ordinary-least-squares weight $\\hat w_j$ by subtracting a constant and cuts it to zero if the constant is larger than its size, $w_j = \\mathrm{sign}(\\hat w_j)\\,\\max(|\\hat w_j| - \\lambda, 0)$, while the ridge (L2, Hoerl and Kennard, 1970) divides every weight by the same factor, $w_j = \\hat w_j / (1 + \\lambda)$, and never makes one exactly zero. The logistic blend is not that case, but the behaviour carries over: L1 selects, L2 spreads. A toy shows what that is worth. In it 3 of 23 members carry a signal and 20 are pure noise; the blend is fitted on 400 compounds and scored on 200 others, 60 times over, with the strength chosen as the stack chooses it.
"""),
    code(STACK_TOY),
    md("""
The L1 blend scores {{=toy_values.loc['L1-penalised blend', 'mean test AUC']:.3f}} and the L2 blend {{=toy_values.loc['L2-penalised blend', 'mean test AUC']:.3f}}; the best single member, chosen on the training part, scores {{=toy_values.loc['best single member', 'mean test AUC']:.3f}}. The difference between the penalties is in the weights: L2 gives a non-zero weight to {{=toy_values.loc['L2-penalised blend', 'mean number of noise members with a non-zero weight (of 20)']:.0f}} of the 20 noise members, L1 to {{=toy_values.loc['L1-penalised blend', 'mean number of noise members with a non-zero weight (of 20)']:.1f}} on average. The unit tests of the module (`tests/test_nested.py`) hold the same lesson in a harder form: with many candidates and few rows an L2 blend spreads weight over noise and can lose to picking the best one, and letting the data choose sparsity removes the need for a hand-chosen "top k". On real data the cell below fits the stack for the first outer fold of the first repeat of CYP3A4 and compares it with the best single member (chosen on the inner scores, as a user would) and with the average of the members' ranks, on the outer-validation compounds, with the plain AUC.
"""),
    code(STACK_REAL),
    md("""
On this fold the cross-validated AUC that the blend saw inside the training part is {{=stack_values['inner AUC of the blend (cross-validated inside the training part)']:.4f}}; on the outer-validation compounds it earns {{=stack_values['outer AUC of the blend']:.4f}}, against {{=stack_values['outer AUC of the best single member (chosen on the inner scores)']:.4f}} for the best single member and {{=stack_values['outer AUC of the average of the ranks']:.4f}} for the average of ranks. One fold is one draw, and the statement that the blend is better is made in 8.7 over the 15 folds with a corrected test. The five weights of largest size are in the second table, and two things about them are worth reading. Some are negative: the members `lgbm_reg_shift_A` and `lgbm_A_deep_knnfeat` have negative weights next to positive weights on close relatives. A blend may subtract a member that shares an error with another, so that the common error cancels and what is left is the part that differs; the weights of a stack are therefore not "importance", and a negative weight does not say that the member is bad. And the intercept and the scale of the weights are those of a logistic fit to the *rank* positions of the members, so that their size says nothing about the members' scales.

*On what actually ran.* `StackedLogit` in `cypmode/challenge/nested.py` is the meta-learner (`_EmpiricalCdf` for the common scale, `_logit` for the log-odds, `_model` for the solver and its settings, `fit` for the five steps); `BestSingle`, `OneStandardError` and `MeanOfTop` are the selectors it is compared with; `nested_evaluate` refits it inside every outer fold. The cell above calls the module's own class, and the toy is computed in the notebook.
"""),
    md("""
## 8.3 A reproducibility bug in the stack: a solver that stops where it happens to be

> **Field:** Software engineering · Statistics

*Where this section sits:* the blend of 8.2 is the solution of a convex problem, which has one optimum, so the same code run twice on the same data should give the same blend. It did not. This section tells how that was seen, what the cause was, and why it mattered, because it is the clearest example in the project of a defect that no test of correctness would have found and that the reproducibility checks of Part XII are built to find.

**What was seen.** The audit of the project (section 12.2) generates its report twice and compares the two byte by byte, and the report recomputes the nested results of the stack. For CYP2D6 the stack gave a nested outer AUC of 0.7527 in one process and 0.7536 in another, and an MCC of the calls of 0.191 in one and 0.204 in the other (the two pairs of values are recorded in a comment of `StackedLogit._model`). The data, the code and the seeds of the cross-validation were the same. The difference of the MCC, 0.013, is larger than most of the differences that the gates of Part IX test.

**What it was.** The solver for the penalised logistic regression, `liblinear`, is a *coordinate descent*: it improves one weight at a time with the others held fixed, sweeping over the weights until the largest improvement of a sweep is smaller than a *tolerance*. It visits the weights in a random order. The problem is convex, so every order leads to the same optimum if the descent is run long enough; but it is stopped when the improvement falls below the tolerance, and the default tolerance, $10^{-4}$, is loose enough that the point at which it stops depends on the order. All the stopping points are near the optimum, within a tolerance-sized neighbourhood, and not at it. A ranking of 750 compounds is sensitive to a neighbourhood of that size, because the compounds near the boundary of the calls are close in score and a tiny change in the weights reorders them. The random order itself came from the global random state of numpy, which differs from one process to the next, so that two processes fitted slightly different stacks.

**The fix, and a demonstration.** Two changes, both in `StackedLogit._model`: the random order is seeded (`random_state`), and the tolerance is tightened to $10^{-9}$ (`META_TOL`), with the iteration limit raised so that it can be reached (`META_MAX_ITER`). The cell reproduces the defect on the inner scores of the first outer fold of CYP2D6: it fits the same L1 blend ten times with ten different random orders, at the default tolerance and at $10^{-9}$, and compares the blends, the AUCs and the MCC of the calls.
"""),
    code(LIBLINEAR_BUG),
    md("""
At the default tolerance the ten random orders give {{=bug_default['different blends (to 6 decimals)']}} different blends (to six decimals), with outer AUCs from {{=bug_default['outer AUC, lowest over 10 orders']:.5f}} to {{=bug_default['outer AUC, highest']:.5f}}, a spread of {{=bug_default['outer AUC, highest'] - bug_default['outer AUC, lowest over 10 orders']:.5f}} in one fold, and {{=bug_default['compounds called by some orders and not by others']}} of the {{=bug_default['of the validation compounds']}} validation compounds are called positive by some of the ten blends and not by the others. At the tolerance of $10^{-9}$ there are {{=bug_tight['different blends (to 6 decimals)']}} blend, the AUC no longer moves, and {{=bug_tight['compounds called by some orders and not by others']}} compounds change their call. In the audit the effect was larger than here because the call rate is chosen from the inner scores of the stack and so also moves with the weights, in every one of the 15 folds. The module has two tests for this: one that two processes with different global random states fit the same stack (`test_stacked_logit_does_not_depend_on_the_global_random_state`), and one that the meta-learner reaches the same optimum whatever the solver seed (`test_the_meta_learner_reaches_the_same_optimum_whatever_the_solver_seed`), which is a regression test of the tolerance. The second is the one that would have failed on the first day.

The lesson has a general form. A randomised algorithm that is run until it is *close enough* gives an answer that depends on the random numbers it used, even when the problem has only one answer. If the answer feeds a decision that is sensitive to small changes, such as a ranking cut at a threshold, "close enough" is a statement about the decision and not only about the optimum. The remedies are to remove the randomness (seed it), to tighten the stopping rule until the answers agree to a precision below the sensitivity of the decision, and to test that two runs agree, which is what a byte-for-byte comparison of the outputs does.

*On what actually ran.* The fix is in `cypmode/challenge/nested.py` (`META_TOL`, `META_MAX_ITER`, `StackedLogit._model`, with the comment that records the two values observed), the tests in `tests/test_nested.py`, and the check that found it, the double run of the audit (`scripts/run_audit.sh`, job 900). The cell above uses the same solver with its two settings for the two tolerances.
"""),
]
