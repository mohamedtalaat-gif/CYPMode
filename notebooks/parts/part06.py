"""Part VI (first part): why a score on the training data lies (6.1), what choosing does to a cross-validated score (6.2), and the nested construction that removes it (6.3)."""

from notebooks.notebook_builder import code, md

TRAINING_SCORE = '''
# The data of this part: the CYP3A4 compounds with both arms measured (section 2.3), and the feature set "A" of section 4.1 (Morgan counts and RDKit descriptors). A model is fitted on all of them and scored
# on the same compounds (the score on the training rows), and, by 5-fold cross-validation, on compounds it has not seen. The number of trees is raised step by step: a model with more trees is more flexible.
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from cypmode.challenge.nested_runners import LGBM_PRESETS
from cypmode.challenge.tdi_targets import training_view

train_file = pd.read_csv(ROOT / "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
feature_store = np.load(ROOT / "data/tdi_track/outputs/caches/feature_store.npz", allow_pickle=True)
label_view = training_view(train_file, "CYP3A4").query("population == 'both_arms'")
pool_rows = label_view.index.to_numpy()
y_pool = label_view["label"].astype(int).to_numpy()
X_pool = np.hstack([feature_store["morgan_c2"][pool_rows], feature_store["rdkit_desc"][pool_rows]])
cv_folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X_pool, y_pool))

def lgbm(n_trees):
    return LGBMClassifier(verbose=-1, n_jobs=2, random_state=0, **{**LGBM_PRESETS["deep"], "n_estimators": n_trees})

flexibility_rows = []
for n_trees in (5, 20, 80, 320):
    on_training_rows = roc_auc_score(y_pool, lgbm(n_trees).fit(X_pool, y_pool).predict_proba(X_pool)[:, 1])
    held_out = np.zeros(len(y_pool))
    for fit_rows, held in cv_folds:
        held_out[held] = lgbm(n_trees).fit(X_pool[fit_rows], y_pool[fit_rows]).predict_proba(X_pool[held])[:, 1]
    flexibility_rows.append({"trees": n_trees, "AUC on the rows it was fitted on": on_training_rows, "AUC on rows it has not seen (5-fold)": roc_auc_score(y_pool, held_out)})
flexibility = pd.DataFrame(flexibility_rows).assign(**{"gap": lambda t: t.iloc[:, 1] - t.iloc[:, 2]})
show_table(flexibility)

# The extreme case, one nearest neighbour on the fingerprint bits: every compound is its own nearest neighbour, so the score on the rows it was fitted on is perfect whatever the chemistry.
bits_pool = feature_store["morgan_bits"][pool_rows].astype(bool)
nearest = KNeighborsClassifier(n_neighbors=1, metric="jaccard", algorithm="brute")
auc_nn_training = roc_auc_score(y_pool, nearest.fit(bits_pool, y_pool).predict_proba(bits_pool)[:, 1])
nn_held_out = np.zeros(len(y_pool))
for fit_rows, held in cv_folds:
    nn_held_out[held] = KNeighborsClassifier(n_neighbors=1, metric="jaccard", algorithm="brute").fit(bits_pool[fit_rows], y_pool[fit_rows]).predict_proba(bits_pool[held])[:, 1]
auc_nn_cv = roc_auc_score(y_pool, nn_held_out)
print(f"one nearest neighbour: AUC on its own training rows {auc_nn_training:.3f}, on unseen rows {auc_nn_cv:.3f}")
n_pool = len(y_pool)
deepest = flexibility.iloc[-1]
shallowest = flexibility.iloc[0]
'''

MAX_OF_NOISE = '''
# What choosing does to a score, on pure noise. m candidates that know nothing (their scores are random numbers, so each has a true AUC of 0.5) are scored on the same compounds; the best of them is
# picked by its score and that score is quoted. The standard deviation of an AUC of 0.5 on n1 positives and n2 negatives follows from the rank test of section 5.4: the variance of U is n1 n2 (N + 1)/12 and the AUC is U/(n1 n2).
from scipy.stats import rankdata

n_pos, n_neg = int(y_pool.sum()), int((1 - y_pool).sum())
sd_noise_auc = float(np.sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg)))
rng = np.random.default_rng(3)

def noise_auc(scores):
    ranks = rankdata(scores, axis=1)
    return (ranks[:, y_pool == 1].sum(axis=1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

max_rows = []
for m in (1, 2, 5, 10, 23, 50, 100):
    best = np.array([noise_auc(rng.normal(size=(m, len(y_pool)))).max() for _ in range(400)])
    one = np.array([noise_auc(rng.normal(size=(1, len(y_pool))))[0] for _ in range(400)])
    max_rows.append({"candidates m": m, "mean score of a single candidate": float(one.mean()), "mean score of the best of m": float(best.mean()), "formula 0.5 + sd * sqrt(2 ln m)": float(0.5 + sd_noise_auc * np.sqrt(2 * np.log(m)))})
show_table(pd.DataFrame(max_rows), floatfmt=".4f")
print(f"standard deviation of the AUC of one random scorer on these compounds: {sd_noise_auc:.4f} (simulated {np.std([noise_auc(rng.normal(size=(1, len(y_pool))))[0] for _ in range(2000)]):.4f})")
best_of_23 = next(r for r in max_rows if r["candidates m"] == 23)
'''

NESTED_REAL = '''
# The same effect on the project's real candidates. The cached predictions of the 23 members of the default stack (section 8.1) on the nested plan of CYP3A4 (3 repeats of 5 outer folds, 4 inner folds): every selector
# is shown the inner out-of-fold predictions of the outer-training rows only, picks a model or a blend, and the frozen choice is scored on the outer-validation rows, which it has never been shown. The score it saw
# (inner) and the score it earns (outer) are reported with their difference, the optimism. Plain AUC, no weights (the weights of 6.4 come later).
from cypmode.challenge.nested import OUTER_REFIT, BestSingle, MeanOfTop, NestedPlan, OneStandardError, PredictionCache, StackedLogit, nested_evaluate
from cypmode.challenge.nested_runners import default_candidates

NESTED_ROOT = ROOT / "data/tdi_track/outputs/nested/CYP3A4"
plan = NestedPlan.load(NESTED_ROOT / "plan.npz")
cache = PredictionCache(NESTED_ROOT / "cache", plan)
sizes_check = {"outer train": len(plan.task(0, 0, OUTER_REFIT).train), "outer validation": len(plan.task(0, 0, OUTER_REFIT).evaluate), "inner train": len(plan.task(0, 0, 0).train), "inner validation": len(plan.task(0, 0, 0).evaluate)}
default_names = [c.name for c in default_candidates()]
selectors = [BestSingle(), OneStandardError(), MeanOfTop(3), MeanOfTop(5), StackedLogit()]
nested_table = nested_evaluate(cache, default_names, selectors)
by_selector = nested_table.groupby("selector")[["inner_score", "outer_score", "optimism"]].mean().loc[[s.name for s in selectors]]
by_selector["spread of the optimism over the 15 folds"] = nested_table.groupby("selector")["optimism"].std().loc[by_selector.index]
show_table(by_selector.reset_index().rename(columns={"selector": "selection procedure", "inner_score": "score it saw (inner)", "outer_score": "score it earned (outer)"}), floatfmt=".4f")

# Candidates on their own: the outer-fold AUC of each member without any selection (every outer-validation row is predicted by a model fitted without it).
own_rows = []
for name in default_names:
    scores = []
    for repeat, outer in plan.outer_folds():
        refit = plan.task(repeat, outer, OUTER_REFIT)
        scores.append(roc_auc_score(plan.labels(refit.evaluate), cache.get(name, refit)))
    own_rows.append({"member": name, "outer AUC": float(np.mean(scores))})
own_table = pd.DataFrame(own_rows).sort_values("outer AUC", ascending=False).reset_index(drop=True)
show_table(own_table.head(6), floatfmt=".4f")

# Why the optimism above is negative. The inner models are fitted on three quarters of the outer-training rows (1400 of 1867 in one fold) and the outer refit on all of them, so even a candidate that nobody selected
# scores a little higher outside than inside: that is the effect of the size of the training set. Averaged over all 23 members and 15 folds it is the "size effect". What remains of the optimism of a selector,
# after the size effect is taken out, is the effect of selecting.
from cypmode.challenge.nested import inner_oof

size_gaps = []
for repeat, outer in plan.outer_folds():
    rows_in, oof_matrix = inner_oof(cache, default_names, repeat, outer)
    y_in = plan.labels(rows_in)
    refit = plan.task(repeat, outer, OUTER_REFIT)
    y_out = plan.labels(refit.evaluate)
    for j, name in enumerate(default_names):
        size_gaps.append(roc_auc_score(y_in, oof_matrix[:, j]) - roc_auc_score(y_out, cache.get(name, refit)))
size_effect = float(np.mean(size_gaps))

# How alike the members are: the rank correlation between the scores of two members on the same rows, averaged over all pairs, and over the pairs of LightGBM classifiers only.
from scipy.stats import spearmanr

rows_0, oof_0 = inner_oof(cache, default_names, 0, 0)
rank_corr = spearmanr(oof_0).statistic
mean_pair_correlation = float(rank_corr[np.triu_indices(len(default_names), 1)].mean())
lgbm_idx = [i for i, n in enumerate(default_names) if n.startswith('lgbm_') and '_reg_' not in n]
mean_lgbm_correlation = float(rank_corr[np.ix_(lgbm_idx, lgbm_idx)][np.triu_indices(len(lgbm_idx), 1)].mean())
selection_effect = by_selector["optimism"] - size_effect
print(f"size effect (inner minus outer, averaged over all members and folds) {size_effect:+.4f}; selection effect of each procedure: " + ", ".join(f"{k} {v:+.4f}" for k, v in selection_effect.items()))
best_single_row = by_selector.loc["best_single"]
stack_row = by_selector.loc["stack_logit"]
best_single_selection_effect, stack_selection_effect = float(selection_effect["best_single"]), float(selection_effect["stack_logit"])
best_member_outer = float(own_table.iloc[0]["outer AUC"])
worst_member_outer = float(own_table.iloc[-1]["outer AUC"])
n_default_members = len(default_names)
n_outer_folds = len(plan.outer_folds())
'''

PLAN_TOY = '''
# The plan on a toy of 24 compounds (4 outer folds, 3 inner folds, one repeat), drawn for the first outer fold: every compound is one cell. Orange: kept back for the outer score of this fold. The three rows below
# it show the three inner tasks; blue cells are the rows a model is fitted on, green cells the rows it predicts. A compound is never in an outer-validation role and in an inner task of the same outer fold.
from cypmode.challenge.nested import NestedPlan as Plan

toy = Plan(pool=np.arange(24), y=np.array([0, 1] * 12), repeats=1, outer_splits=4, inner_splits=3, seed=0)
outer_refit = toy.task(0, 0, OUTER_REFIT)
grid = np.zeros((4, 24))
grid[0, outer_refit.evaluate] = 3                                     # kept back
for inner in range(3):
    task = toy.task(0, 0, inner)
    grid[1 + inner, task.train] = 1
    grid[1 + inner, task.evaluate] = 2
    grid[1 + inner, outer_refit.evaluate] = 3
from matplotlib.colors import ListedColormap

fig, axis = plt.subplots(figsize=(9.2, 2.6))
axis.imshow(grid, cmap=ListedColormap([theme["background"], theme["blue"], theme["green"], theme["orange"]]), aspect="auto", vmin=0, vmax=3)
axis.set_yticks(range(4), ["outer fold 0", "inner task 0", "inner task 1", "inner task 2"])
axis.set_xticks(range(24), [str(i) for i in range(24)], fontsize=7)
axis.set_xlabel("compound (toy pool of 24)")
tidy(axis, theme)
fig.tight_layout()
toy_inner_sizes = [(len(toy.task(0, 0, i).train), len(toy.task(0, 0, i).evaluate)) for i in range(3)]
toy_outer_size = len(outer_refit.evaluate)
'''

PLAN_CHECKS = '''
# The real plan, checked by code. For every outer fold: (a) the inner tasks together predict every outer-training row exactly once, (b) no inner task touches an outer-validation row, (c) the outer refit is fitted on
# exactly the outer-training rows. Then the size of the plan, and the identity rule: asking for more repeats must not change the plan, so that earlier predictions are reused.
problems = []
for repeat, outer in plan.outer_folds():
    refit = plan.task(repeat, outer, OUTER_REFIT)
    inner_tasks = plan.inner_tasks(repeat, outer)
    predicted = np.concatenate([t.evaluate for t in inner_tasks])
    touched = np.unique(np.concatenate([np.concatenate([t.train, t.evaluate]) for t in inner_tasks]))
    problems += [("a", repeat, outer)] * (len(predicted) != len(set(predicted.tolist())) or set(predicted.tolist()) != set(refit.train.tolist()))
    problems += [("b", repeat, outer)] * bool(set(touched.tolist()) & set(refit.evaluate.tolist()))
kinds = pd.Series(["inner" if t.inner >= 0 else ("outer refit" if t.inner == OUTER_REFIT else ("deployment fit" if t.inner == -2 else "other")) for t in plan.tasks()]).value_counts()
tasks_per_repeat = int(len([t for t in plan.tasks() if t.repeat == 0]))
fewer_repeats = NestedPlan(plan.pool, plan.y, 1, plan.outer_splits, plan.inner_splits, plan.seed, plan.external)
print(f"pool {len(plan.pool)} compounds; {plan.repeats} repeats x {plan.outer_splits} outer folds x {plan.inner_splits} inner folds; tasks: {({k: int(v) for k, v in kinds.items()})}; {tasks_per_repeat} per repeat; problems found: {problems}")
print(f"plan id {plan.plan_id}; with one repeat instead of three: {fewer_repeats.plan_id}; the same: {fewer_repeats.plan_id == plan.plan_id}")
files_per_member = len(list((NESTED_ROOT / "cache" / plan.plan_id / default_names[0]).glob("*.npy")))
n_members_cached = len(cache.candidates())
n_plan_tasks = len(plan.tasks())
print("sizes of one outer fold:", sizes_check)
'''

CELLS = [
    md("""
# Part VI. Measuring honestly

Parts II to IV gave the data, the numbers that describe a molecule and the models that learn from them. Parts VII to XI will report dozens of experiments, and every one of them is a claim of the same form: *this model, or this addition to a model, is better than that one.* A claim of that form is worth exactly as much as the measurement behind it, and a measurement that flatters the model is worse than none, because it is believed.
This part builds the measurement that every later number rests on. It begins with the two ways in which a score is inflated without anyone intending it (6.1 and 6.2), and then the construction that removes the inflation, **nested cross-validation** (6.3). Sections 6.4 and 6.5 make the measurement fit the challenge and make a comparison of two models honest. Section 6.6 tells why an earlier analysis of the project was dropped and redone,
section 6.7 turns the rules into pre-declared gates and section 6.8 describes the checks that a split is informative and that nothing leaks.

What this part hands on: the nested plan and its cache of predictions are the objects that the final TDI model of Part VIII is built from; the corrected test of 6.5 and the gate of 6.7 are the judges of every addition in Parts VII, IX and X; the hold-out of 6.8 is the second opinion that the final stack is checked against. A reader who wants only to follow the later parts can read 6.3, 6.5 and 6.7, which the later parts cite by number, and return to the rest when it is needed. The data of this part are, throughout, those of
CYP3A4 (the compounds with both arms measured, section 2.3); the same code runs for CYP2D6 and its results are given in Part VIII.
"""),
    md("""
## 6.1 Why a score on the training data lies

> **Field:** Statistics · Machine learning

*Where this section sits:* Part IV ended with the metrics that judge a model (section 4.10). This section asks on which compounds the metric should be computed, and the answer, which looks obvious once seen, is the foundation of the part: not on the compounds the model was fitted on.

A model that has been fitted to a set of compounds can do two things with them. It can learn what is general, the relation between structure and label that will hold for compounds it has never seen, and it can memorise what is particular, the individual compounds and their labels, including the noise in the labels. The score on the compounds it was fitted on, which this notebook calls the **score on the fitted rows** (the literature says *resubstitution* or *training score*), rewards both. The score on compounds it has never seen rewards only the first.
The difference between the two is the **optimism** of the training score, $O = S_{\\text{fitted}} - S_{\\text{new}}$, where $S_{\\text{fitted}}$ is a score (here the AUC of section 4.10) on the rows used for fitting and $S_{\\text{new}}$ the same score on rows that were not. It is never negative in expectation, and it grows with the flexibility of the model: the more a model can bend to the data, the more of the data's accident it absorbs.

The cell shows this on the project's data, with a model of increasing flexibility: the deep LightGBM of section 4.3 with 5, 20, 80 and 320 trees. For each size it reports the AUC on the rows it was fitted on and the AUC on rows it has not seen, obtained by 5-fold cross-validation (section 4.1: five parts, each held out in turn while the model is fitted on the other four). It then adds the extreme case, one nearest neighbour on the fingerprints (section 4.5), which memorises by construction.
"""),
    code(TRAINING_SCORE),
    md("""
Every number in the second column is high and none of them can be trusted. With {{=int(shallowest['trees'])}} trees the model scores {{=shallowest['AUC on the rows it was fitted on']:.3f}} on the rows it was fitted on and {{=shallowest['AUC on rows it has not seen (5-fold)']:.3f}} on rows it has not seen; with {{=int(deepest['trees'])}} trees it scores {{=deepest['AUC on the rows it was fitted on']:.3f}}, a perfect ranking of the {{=n_pool:,}} compounds it was fitted on, and {{=deepest['AUC on rows it has not seen (5-fold)']:.3f}} on new ones. The gap, the optimism,
grows from {{=shallowest['gap']:.2f}} to {{=deepest['gap']:.2f}} as the model gets more flexible, while the score on new rows improves only from {{=shallowest['AUC on rows it has not seen (5-fold)']:.2f}} to {{=deepest['AUC on rows it has not seen (5-fold)']:.2f}}. The nearest-neighbour model is the limiting case: each compound is its own nearest neighbour, so its AUC on the rows it was fitted on is {{=auc_nn_training:.3f}} whatever the chemistry, and on unseen rows it is {{=auc_nn_cv:.3f}}, barely above the 0.5 of a coin.
Two models with the same score on the fitted rows (the nearest-neighbour model and the 320-tree model both score a perfect {{=auc_nn_training:.1f}}) can differ enormously in what they know.

The lesson is a rule that the whole project follows: **a score is computed only on rows that took no part in fitting the model, and in making any choice about it.** The first half of the rule is what cross-validation provides. The second half, "and in making any choice about it", is the subject of the next section, because cross-validation by itself does not satisfy it.
"""),
    md("""
## 6.2 Cross-validation, and what happens to its score when it is also used to choose

> **Field:** Statistics · Machine learning

*Where this section sits:* section 6.1 established that a score must be computed on rows the model has not seen, and cross-validation does that. But the project does not fit one model: it fits dozens (section 8.1), and picks among them, and a cross-validation score that was used to pick the winner is no longer a clean measurement of the winner. This section shows why, with an example in which the truth is known, and then measures the size of the effect on the project's real candidates, where it turns out to be small and to be hidden behind a larger effect of a different kind.

**The maximum of noisy numbers.** A cross-validated score is an estimate, and an estimate has noise: another partition of the compounds, or another sample of compounds from the same population, would give a slightly different number. Suppose there are $m$ candidates and the true quality of each is the same, $\\mu$, and each estimate has noise of standard deviation $\\sigma$. The estimates are then $m$ draws from a bell curve around $\\mu$. If the best estimate is quoted as the quality of the winner, the quoted value is the largest of $m$ draws,
and the largest of several draws is higher than the average draw: for many candidates it is about

$$\\mathbb{E}\\bigl[\\max\\bigr] \\;\\approx\\; \\mu + \\sigma\\sqrt{2\\ln m},$$

where $\\mathbb{E}$ is the expected value and $\\ln$ the natural logarithm. The quoted score is too high by $\\sigma\\sqrt{2\\ln m}$ although each single estimate was unbiased, and the bias grows, slowly, with the number of candidates tried. The cell makes this concrete with candidates that know nothing: each one assigns random numbers as scores, so its true AUC is 0.5 exactly, and the AUC of one random scorer on these compounds has the standard deviation that follows from
the rank test of section 5.4 (the variance of the statistic $U$ is $n_1 n_2 (N+1)/12$ and the AUC is $U/(n_1 n_2)$, with $n_1$ positives, $n_2$ negatives and $N = n_1 + n_2$ compounds). The best of $m$ such scorers is picked by its score and the score is read off.
"""),
    code(MAX_OF_NOISE),
    md("""
A single random scorer scores {{=max_rows[0]['mean score of a single candidate']:.3f}} on average, as it must, with a standard deviation of {{=sd_noise_auc:.4f}}. The best of 23 of them scores {{=best_of_23['mean score of the best of m']:.3f}} on average, though not one of them knows anything, and the best of 100 scores {{=max_rows[-1]['mean score of the best of m']:.3f}}. The formula over-predicts a little for small $m$ (it is the form for many candidates) but has the right shape: slow growth with the number of candidates. The effect in this example is a few hundredths of AUC, and in the project's
own work the differences that matter are of that size or smaller (section 6.7 puts numbers on it): a difference between two candidates of 0.01 in AUC is, by this arithmetic, not larger than what picking the best of a few dozen candidates can invent. So the project does not take its numbers from the contest that chose the winner.

**On the real candidates.** The honest way to measure a *procedure* that picks a model is to let the procedure see only part of the compounds and to score its pick on the compounds it did not see. The project's predictions are stored for exactly this (section 6.3), and the cell uses them for the 23 members of the default stack. Each *selection procedure* is shown the cross-validated predictions of the members on the training rows of an outer fold only (the *inner* score, the one it picked from), picks a member or a blend of members
(five procedures: the best single member; the simplest member within one standard error of the best; the average of the best three, or five, members; and the logistic stack of section 8.2), and the frozen pick is scored on the held-out outer rows, which it has never been shown (the *outer* score). The two scores and their difference, inner minus outer, are reported. Its sign is the surprise.
"""),
    code(NESTED_REAL),
    md("""
The inner score of the best single member is {{=best_single_row['inner_score']:.4f}} and its outer score {{=best_single_row['outer_score']:.4f}}: the optimism, inner minus outer, is {{=best_single_row['optimism']:+.4f}}, **negative**, and the same holds for every procedure, from {{=by_selector['optimism'].max():+.4f}} to {{=by_selector['optimism'].min():+.4f}}. The inner estimate is pessimistic, not optimistic. The reason is a second effect, of a different kind, that works in the opposite direction. An inner model is fitted on three quarters of the outer-training rows ({{=sizes_check['inner train']:,}}
of {{=sizes_check['outer train']:,}} in the first fold) and the outer refit on all of them, so even a member that nobody selected scores a little higher outside than inside, because it had more compounds to learn from. Averaged over all {{=n_default_members}} members and {{=n_outer_folds}} folds, that **size effect** is {{=size_effect:+.4f}} AUC. Take it out of the optimism of each procedure and what remains is the effect of selecting, the one of the example above:
$S_{\\text{in}} - S_{\\text{out}} = (\\text{size effect}) + (\\text{selection effect})$, with $S_{\\text{in}}$ and $S_{\\text{out}}$ the inner and the outer score. For the best single member the selection effect is {{=best_single_selection_effect:+.4f}}, and for the logistic stack, whose weights are fitted on the inner predictions, it is {{=stack_selection_effect:+.4f}}: both are smaller than {{=max(abs(best_single_selection_effect), abs(stack_selection_effect)):.3f}} in absolute value, far below the {{=by_selector['spread of the optimism over the 15 folds'].min():.3f}} to
{{=by_selector['spread of the optimism over the 15 folds'].max():.3f}} by which the difference varies from fold to fold.

Why is the effect of selecting so small here, when the example of noise made it large? Because the candidates of the example were independent and these are not. Most of the 23 members are boosted trees on overlapping sets of features and make nearly the same errors: the average rank correlation between the scores of two members on the same rows is {{=mean_pair_correlation:.2f}}, and {{=mean_lgbm_correlation:.2f}} between two of the LightGBM classifiers. Estimates that move together have a maximum that is hardly above the typical one, so the bias of selecting grows with the number of *independent* candidates and not with the number of names in a list.
Their own outer AUCs, with no selection involved, run from {{=worst_member_outer:.3f}} to {{=best_member_outer:.3f}}, and the best few are within a few thousandths of each other, so even a wrong pick among them costs little. The bias was worth measuring because it could not be assumed small, and it is not large here; it would be larger for a set of genuinely different candidates of the same quality. The logistic stack earns an outer AUC of {{=stack_row['outer_score']:.4f}}, against {{=best_member_outer:.4f}} for the best member on its own.

Two practical rules follow, and the project keeps both. A score is never quoted from the contest that produced the choice: the quoted number is the outer score of the whole procedure, choice included. And the inner and the outer score are not on the same footing, because the training sets differ in size, so they are never compared with each other as if a difference between them were optimism: only outer scores are compared with outer scores (6.5).
"""),
    md("""
## 6.3 Nested cross-validation: the construction, every loop, what is fitted where

> **Field:** Statistics · Machine learning · Software engineering

*Where this section sits:* sections 6.1 and 6.2 said what must be avoided. This section builds the evaluation that avoids it, the one that every number from Part VII on is computed with, and describes the objects of the code (`cypmode/challenge/nested.py`) that hold it. The final model of Part VIII is built from the same objects, so a reader who understands this section understands how the model was made as well as how it was judged.

**The construction.** Cut the training compounds into $K$ disjoint parts, the **outer folds**. Hold one outer fold out: it is the *outer-validation* set, kept for scoring and never used for any choice. The remaining rows are the *outer-training* set. Cut that set, again, into $M$ parts, the **inner folds**. Everything that involves a choice, every candidate's fit, the picking of a member, the weights of a blend, the operating threshold, happens inside the outer-training set, with the inner folds supplying the out-of-fold predictions on
which the choices are made (a model predicts each inner fold after being fitted on the others, so every outer-training row gets a prediction from a model that did not see it). When the choices are frozen, the whole procedure is fitted once more on the entire outer-training set and its pick is scored on the outer-validation set. That is one outer score. Repeat it for each of the $K$ outer folds, and again for $R$ different random partitions (the **repeats**), and the $J = KR$ outer scores are the evaluation of the procedure. With $T_j$ the
outer-training rows of the $j$-th outer fold, $V_j$ its outer-validation rows and $\\Pi(T_j)$ the whole procedure (every choice) fitted using $T_j$ alone, and $S(V_j ;\\ \\Pi(T_j))$ the score (for example the AUC) of its predictions for $V_j$,

$$\\hat S_{\\text{nested}} \\;=\\; \\frac{1}{J} \\sum_{j=1}^{J} S\\bigl(V_j ;\\ \\Pi(T_j)\\bigr).$$

The project uses $K = 5$, $M = 4$ and $R = 3$ (and, for confirmation, $K = 10$, section 6.7). The plan is a plain table of index sets: a **task** says "fit a model on these rows and predict those rows", and a prediction is the output of one task for one candidate. Because the plan is data and not code, any program in any environment can carry out a task (LightGBM here, a neural network in another process, a message-passing network in a third; section 12.4 explains why they must not share a process) and
write its predictions into a **prediction cache**, one file for each combination of candidate and task, named after the plan's identity. A selector then reads only the inner predictions of an outer fold, and `nested_evaluate` scores what it chose on the outer rows.

A toy makes the geometry clear: 24 compounds, 4 outer folds and 3 inner folds, drawn for the first outer fold. The cells of the first row are the rows kept back for the outer score; in each of the three rows below it, the blue cells are the rows a model is fitted on and the green cells the rows it predicts, and the orange cells, always the same, are the held-out rows that never appear as blue or green.
"""),
    code(PLAN_TOY),
    md("""
In the toy the outer-validation set has {{=toy_outer_size}} compounds, and the three inner tasks fit on {{=toy_inner_sizes[0][0]}} and predict {{=toy_inner_sizes[0][1]}} rows each, all of them among the other 18. A figure can show that the orange cells are never blue or green in this one fold; the next cell shows it by *testing* it, on the real plan of CYP3A4, for every one of the {{=n_outer_folds}} outer folds, and reports the size of the plan.
"""),
    code(PLAN_CHECKS),
    md("""
The real plan covers {{=len(plan.pool):,}} compounds and has {{=n_plan_tasks}} tasks: {{=tasks_per_repeat}} for each of the {{=plan.repeats}} repeats (five outer folds, each with four inner tasks and one outer refit) and a last set for the deployment (four inner tasks on the whole pool and the fit of the whole pool, which predicts the blinded test compounds). The three checks of the cell found {{=len(problems)}} problems: in every outer fold the inner tasks together predict each outer-training row exactly once, no inner task touches a row of the outer-validation set, and the outer refit is fitted on
exactly the outer-training rows. One outer fold has {{=sizes_check['outer train']:,}} rows to fit on and {{=sizes_check['outer validation']}} to score; an inner task fits on {{=sizes_check['inner train']:,}} and predicts {{=sizes_check['inner validation']}}, which is the size effect of 6.2 in numbers. The identity of the plan, {{=plan.plan_id}}, does not depend on the number of repeats (the plan with one repeat has the same one), which is deliberate: asking for more repeats later reuses every prediction already stored and computes only the new tasks,
and that is how the confirmatory repeats of 6.7 were added without redoing anything. Every member of the default stack has {{=files_per_member}} files in the cache, one per task, and the cache holds {{=n_members_cached}} members in all, the members of the stack and of every addition that was tested.

**What this buys, and what it costs.** The procedure never lets a choice see the rows that score it, at any level, and its cost is computation: each member has to be fitted for every task, {{=tasks_per_repeat}} fits per repeat instead of one, and a neural member that takes minutes per fit takes days of the machine's time (Part XII describes the queue that makes this affordable). The benefit is that the single number at the end, the outer score of the whole pipeline, is an estimate that can be defended, and that two pipelines can be compared fairly because they are scored on the same outer folds (6.5). It also gives the final build of
Part VIII its structure: the deployment task is exactly an outer fold in which the outer-validation set is the blinded test set.

*On what actually ran.* `NestedPlan` builds the folds (stratified on the label, and optionally keeping whole chemical groups together, section 3.6), saves them as `plan.npz`, and gives them their identity from the pool, the labels and the fold counts. `PredictionCache` stores one `float32` array per candidate and task under `data/tdi_track/outputs/nested/<enzyme>/cache/<plan id>/<candidate>/r<repeat>_k<outer>_m<inner>.npy`. `Selector` and its subclasses (`BestSingle`, `OneStandardError`, `MeanOfTop`, `StackedLogit`) see the inner out-of-fold matrix and return a frozen choice;
`nested_evaluate` scores it on the outer rows and reports the two scores and their difference. `cypmode/challenge/nested_runners.py` fills the cache for the LightGBM members and the module of Part IV for the neural ones; `nested_deploy.py` and `nested_report.py` apply the same selection to the whole pool and print the comparison of members and procedures.
"""),
]
