# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part IV (first part): machine learning from zero. Supervised learning, decision trees, gradient boosting, forests."""

from notebooks.notebook_builder import code, figure, md

ML_DATA = '''
# The working data of this part: the CYP3A4 compounds with both arms (section 2.3), their label, their shift s, and the feature tables of the store (section 3.7), rows in the order of the training file.
# Every model below is judged by 5-fold cross-validation: the compounds are divided into five parts, each part in turn is held out, the model is fitted on the other four and scored on the held-out part.
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

view_ml = training_view(train, "CYP3A4").query("population == 'both_arms'")
rows_ml = view_ml.index.to_numpy()
y_ml = view_ml["label"].astype(int).to_numpy()
shift_ml = view_ml["shift"].to_numpy()
desc_ml, counts_ml = store["rdkit_desc"][rows_ml], store["morgan_c2"][rows_ml]
set_a = np.hstack([counts_ml, desc_ml])                                        # the project's feature set "A": Morgan counts and RDKit descriptors
descriptor_names = [name for name, _ in Descriptors.descList]
folds_ml = list(StratifiedKFold(5, shuffle=True, random_state=0).split(set_a, y_ml))

def cross_validated_auc(fit_predict, features, target=None):
    """Mean and standard deviation over the five held-out parts of the AUC of fit_predict(train_x, train_y, test_x)."""
    target = y_ml if target is None else target
    scores = [roc_auc_score(y_ml[held], fit_predict(features[fit_rows], target[fit_rows], features[held])) for fit_rows, held in folds_ml]
    return float(np.mean(scores)), float(np.std(scores))

print(f"{len(y_ml)} compounds, {int(y_ml.sum())} positive ({y_ml.mean():.1%}); feature set A has {set_a.shape[1]} columns ({counts_ml.shape[1]} Morgan counts and {desc_ml.shape[1]} descriptors)")
n_ml, positives_ml, columns_a = len(y_ml), int(y_ml.sum()), set_a.shape[1]
'''

LOGLOSS = '''
# The log-loss for one compound and its derivative with respect to the logit z, p = 1 / (1 + exp(-z)). The derivation in the text says the derivative is p - y; a finite difference checks it.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y, z):
    p = sigmoid(z)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

step = 1e-6
check_rows = []
for y_value in (0, 1):
    for z in (-3.0, -0.5, 0.0, 2.0):
        numeric = (log_loss(y_value, z + step) - log_loss(y_value, z - step)) / (2 * step)
        check_rows.append({"label y": y_value, "logit z": z, "probability p": sigmoid(z), "loss": log_loss(y_value, z), "derivative p - y": sigmoid(z) - y_value, "finite difference": numeric})
show_table(pd.DataFrame(check_rows), floatfmt=".5f")
worst_gap = max(abs(r["derivative p - y"] - r["finite difference"]) for r in check_rows)
penalty = {p: float(-np.log(p)) for p in (0.9, 0.5, 0.1, 0.01)}
print("loss of a positive compound given probability p:", {p: round(v, 2) for p, v in penalty.items()})
'''

BIAS_VARIANCE = '''
# Bias and variance measured. The truth is g(x) = sin(2 pi x); each training set has 30 points with noise of standard deviation 0.3; a polynomial of a given degree is fitted to 400 independent training
# sets and evaluated on 60 fixed points. Bias squared is how far the average prediction is from the truth, variance is how much the predictions scatter around their own average.
rng = np.random.default_rng(11)
noise_sd, n_points, repetitions = 0.3, 30, 400
grid_x = np.linspace(0.02, 0.98, 60)
truth = np.sin(2 * np.pi * grid_x)
bv_rows = []
for degree in (1, 3, 5, 7):
    predictions = np.empty((repetitions, grid_x.size))
    train_error, test_error = [], []
    for r in range(repetitions):
        x = rng.uniform(0, 1, n_points)
        y = np.sin(2 * np.pi * x) + rng.normal(0, noise_sd, n_points)
        fit = np.polynomial.Polynomial.fit(x, y, degree)
        predictions[r] = fit(grid_x)
        train_error.append(np.mean((y - fit(x)) ** 2))
        test_error.append(np.mean((truth + rng.normal(0, noise_sd, grid_x.size) - predictions[r]) ** 2))
    bias_squared = float(np.mean((predictions.mean(axis=0) - truth) ** 2))
    variance = float(np.mean(predictions.var(axis=0)))
    bv_rows.append({"degree": degree, "error on the training points": float(np.mean(train_error)), "bias squared": bias_squared, "variance": variance, "noise": noise_sd**2,
                    "bias squared + variance + noise": bias_squared + variance + noise_sd**2, "measured error on new points": float(np.mean(test_error))})
show_table(pd.DataFrame(bv_rows), floatfmt=".3f")
decomposition_gap = max(abs(r["bias squared + variance + noise"] - r["measured error on new points"]) for r in bv_rows)
best_degree = min(bv_rows, key=lambda r: r["measured error on new points"])["degree"]
'''

STUMP = '''
# The split search written out. For every descriptor, sort the compounds by its value, and for every position between two different values compute the Gini impurity of the two parts that the split makes,
# with cumulative sums so that all positions cost one pass. The best split of all 217 descriptors is compared with the one scikit-learn's tree of depth 1 chooses.
from sklearn.tree import DecisionTreeClassifier

def gini(positive, total):
    p = positive / total
    return 2 * p * (1 - p)                                                      # 1 - p^2 - (1 - p)^2 for two classes

def best_stump(features, target):
    n = len(target)
    parent = gini(target.sum(), n)
    best = {"gain": -1.0}
    for j in range(features.shape[1]):
        order = np.argsort(features[:, j], kind="stable")
        values, labels = features[order, j], target[order]
        left_n = np.arange(1, n)
        left_positive = np.cumsum(labels)[:-1]
        between = values[1:] > values[:-1]                                      # a split is only possible between two different values
        if not between.any():
            continue
        right_n, right_positive = n - left_n, labels.sum() - left_positive
        gain = np.where(between, parent - (left_n * gini(left_positive, left_n) + right_n * gini(right_positive, right_n)) / n, -1.0)
        k = int(gain.argmax())
        if gain[k] > best["gain"]:
            best = {"gain": float(gain[k]), "feature": j, "threshold": float((values[k] + values[k + 1]) / 2), "left_n": int(left_n[k]), "left_rate": float(left_positive[k] / left_n[k]),
                    "right_n": int(right_n[k]), "right_rate": float(right_positive[k] / right_n[k])}
    return best

mine = best_stump(desc_ml, y_ml)
library = DecisionTreeClassifier(max_depth=1, criterion="gini", random_state=0).fit(desc_ml, y_ml)
node = library.tree_
library_gain = node.impurity[0] - (node.n_node_samples[1] * node.impurity[1] + node.n_node_samples[2] * node.impurity[2]) / node.n_node_samples[0]
stump_rows = [{"": "split search written out here", "descriptor": descriptor_names[mine["feature"]], "threshold": mine["threshold"], "impurity decrease": mine["gain"], "share positive, below": mine["left_rate"], "share positive, above": mine["right_rate"]},
              {"": "scikit-learn, depth 1", "descriptor": descriptor_names[node.feature[0]], "threshold": float(node.threshold[0]), "impurity decrease": float(library_gain),
               "share positive, below": float(node.value[1][0][1] / node.value[1][0].sum()), "share positive, above": float(node.value[2][0][1] / node.value[2][0].sum())}]
show_table(pd.DataFrame(stump_rows), floatfmt=".4f")
parent_gini = gini(y_ml.sum(), len(y_ml))
same_split = mine["feature"] == node.feature[0] and abs(mine["threshold"] - node.threshold[0]) < 1e-6
print(f"impurity of the whole set (Gini) {parent_gini:.4f}; the two searches choose the same split: {same_split}")
'''

TREE_BEHAVIOUR = '''
# How a tree behaves as it is allowed to grow, and how stable its first question is. Left: the AUC on the part a tree was fitted on and on the held-out part, by depth. Right: the descriptor chosen for the
# root of a tree of depth 2 in 60 bootstrap samples of the data (a bootstrap sample is drawn with replacement, so about two thirds of the compounds appear in it), and the three questions at the top.
depth_rows = []
for depth in (1, 2, 3, 5, 8, None):
    fitted, held_scores = [], []
    for fit_rows, held in folds_ml:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(desc_ml[fit_rows], y_ml[fit_rows])
        fitted.append(roc_auc_score(y_ml[fit_rows], tree.predict_proba(desc_ml[fit_rows])[:, 1]))
        held_scores.append(roc_auc_score(y_ml[held], tree.predict_proba(desc_ml[held])[:, 1]))
    depth_rows.append({"depth of the tree": "no limit" if depth is None else depth, "AUC on the compounds it was fitted on": float(np.mean(fitted)), "AUC on held-out compounds": float(np.mean(held_scores))})
show_table(pd.DataFrame(depth_rows), floatfmt=".3f")

rng = np.random.default_rng(5)
roots, shapes = [], []
for _ in range(60):
    sample = rng.integers(0, len(y_ml), len(y_ml))
    nodes = DecisionTreeClassifier(max_depth=2, random_state=0).fit(desc_ml[sample], y_ml[sample]).tree_
    roots.append(descriptor_names[nodes.feature[0]])
    shapes.append(tuple(descriptor_names[nodes.feature[k]] if nodes.feature[k] >= 0 else "leaf" for k in (0, 1, 4)))     # the questions at the root and at its two children
root_counts = pd.Series(roots).value_counts()
shape_counts = pd.Series([" / ".join(s) for s in shapes]).value_counts()
print("the descriptor at the root of a depth-2 tree in 60 bootstrap samples:", {k: int(v) for k, v in root_counts.head(4).items()})
print("the three questions (root, left child, right child) in the same 60 trees:", len(shape_counts), "different combinations; the most frequent, in", int(shape_counts.iloc[0]), "trees:", shape_counts.index[0])
fitted_unlimited, held_unlimited = depth_rows[-1]["AUC on the compounds it was fitted on"], depth_rows[-1]["AUC on held-out compounds"]
best_depth_row = max(depth_rows, key=lambda r: r["AUC on held-out compounds"])
n_distinct_roots, root_top_share = len(root_counts), float(root_counts.iloc[0] / root_counts.sum())
n_shapes, shape_top_share = len(shape_counts), float(shape_counts.iloc[0] / shape_counts.sum())
'''

NEWTON = '''
# Second-order gradient boosting written out in about forty lines, with trees built from histograms of the gradient statistics. Each round computes, for every compound, the gradient g = p - y and the
# curvature h = p (1 - p) of the log-loss at the current logit; grows a tree of depth 3 by the gain formula of the text; sets each leaf to -G / (H + lambda); and adds eta times the leaf values to the logits.
class NewtonBooster:
    def __init__(self, rounds=100, eta=0.1, depth=3, lam=1.0, gamma=0.0, min_hessian=1.0, bins=32):
        self.rounds, self.eta, self.depth, self.lam, self.gamma, self.min_hessian, self.bins = rounds, eta, depth, lam, gamma, min_hessian, bins

    def _codes(self, X):
        return np.column_stack([np.searchsorted(self.edges[j], X[:, j], side="right") for j in range(X.shape[1])]).astype(np.int64)

    def _grow(self, codes, g, h, idx, depth):
        G, H = g[idx].sum(), h[idx].sum()
        leaf = -G / (H + self.lam)
        if depth == 0 or len(idx) < 2:
            return leaf
        n_features, width = codes.shape[1], self.bins
        flat = (codes[idx] + (np.arange(n_features) * width)[None, :]).ravel()
        g_hist = np.bincount(flat, weights=np.repeat(g[idx], n_features), minlength=n_features * width).reshape(n_features, width)
        h_hist = np.bincount(flat, weights=np.repeat(h[idx], n_features), minlength=n_features * width).reshape(n_features, width)
        g_left, h_left = np.cumsum(g_hist, axis=1)[:, :-1], np.cumsum(h_hist, axis=1)[:, :-1]
        g_right, h_right = G - g_left, H - h_left
        with np.errstate(divide="ignore", invalid="ignore"):                       # an empty side with lambda = 0 divides 0 by 0: such a split has no gain
            gain = 0.5 * (g_left**2 / (h_left + self.lam) + g_right**2 / (h_right + self.lam) - G**2 / (H + self.lam)) - self.gamma
        gain[np.isnan(gain) | (h_left < self.min_hessian) | (h_right < self.min_hessian)] = -np.inf
        f, b = np.unravel_index(np.argmax(gain), gain.shape)
        if not np.isfinite(gain[f, b]) or gain[f, b] <= 0:
            return leaf
        goes_left = codes[idx, f] <= b
        return (f, b, self._grow(codes, g, h, idx[goes_left], depth - 1), self._grow(codes, g, h, idx[~goes_left], depth - 1), float(gain[f, b]))

    def _apply(self, node, codes, idx, out):
        if not isinstance(node, tuple):
            out[idx] = node
            return
        f, b, left, right, _ = node
        goes_left = codes[idx, f] <= b
        self._apply(left, codes, idx[goes_left], out)
        self._apply(right, codes, idx[~goes_left], out)

    def _tree_values(self, tree, codes):
        out = np.zeros(len(codes))
        self._apply(tree, codes, np.arange(len(codes)), out)
        return out

    def fit(self, X, y, X_watch=None, y_watch=None, watch_every=10):
        self.edges = [np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.bins + 1)[1:-1])) for j in range(X.shape[1])]
        codes = self._codes(X)
        watch_codes = self._codes(X_watch) if X_watch is not None else None
        rate = y.mean()
        self.base = float(np.log(rate / (1 - rate)))
        logit, watch_logit = np.full(len(y), self.base), (np.full(len(X_watch), self.base) if X_watch is not None else None)
        self.trees, self.history = [], []
        for round_number in range(1, self.rounds + 1):
            p = sigmoid(logit)
            tree = self._grow(codes, p - y, p * (1 - p), np.arange(len(y)), self.depth)
            self.trees.append(tree)
            logit = logit + self.eta * self._tree_values(tree, codes)
            if X_watch is not None:
                watch_logit = watch_logit + self.eta * self._tree_values(tree, watch_codes)
                if round_number % watch_every == 0:
                    self.history.append({"round": round_number, "AUC on the fitted compounds": roc_auc_score(y, logit), "AUC on the watched compounds": roc_auc_score(y_watch, watch_logit)})
        return self

    def decision_function(self, X):
        codes = self._codes(X)
        return self.base + self.eta * sum(self._tree_values(tree, codes) for tree in self.trees)

# Round one checked against the theory: the second-order objective predicts how much one tree of depth 1 lowers the summed log-loss (it is the gain of the best split); the true change is measured.
p0 = np.full(len(y_ml), y_ml.mean())
g0, h0 = p0 - y_ml, p0 * (1 - p0)
base_logit = np.log(p0[0] / (1 - p0[0]))
one = NewtonBooster(rounds=1, eta=1.0, depth=1, lam=0.0, min_hessian=1.0)
one.fit(desc_ml, y_ml)
predicted_decrease = one.trees[0][4]
after = one.decision_function(desc_ml)
actual_decrease = float(log_loss(y_ml, np.full(len(y_ml), base_logit)).sum() - log_loss(y_ml, after).sum())
print(f"first round, one split, lambda = 0: the formula predicts a decrease of {predicted_decrease:.2f} in the summed log-loss; the true decrease is {actual_decrease:.2f} (from {log_loss(y_ml, np.full(len(y_ml), base_logit)).sum():.1f})")
'''

NEWTON_CV = '''
# The written-out booster against LightGBM with matching settings, in the same five folds, and the AUC of the booster on the fitted and on the held-out compounds as rounds are added (first fold), for two
# values of the learning rate eta.
from lightgbm import LGBMClassifier

def boost_fit_predict(rounds, eta):
    return lambda train_x, train_y, test_x: NewtonBooster(rounds=rounds, eta=eta, depth=3, lam=1.0).fit(train_x, train_y).decision_function(test_x)

def lightgbm_fit_predict(train_x, train_y, test_x):
    model = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=8, max_depth=3, reg_lambda=1.0, min_child_samples=5, n_jobs=2, verbose=-1, random_state=0)
    return model.fit(train_x, train_y).predict_proba(test_x)[:, 1]

import time
timing = {}
compare = []
for name, function in (("the booster written out above (100 rounds, eta 0.1, depth 3)", boost_fit_predict(100, 0.1)), ("LightGBM with matching settings", lightgbm_fit_predict)):
    started = time.time()
    mean_auc, sd_auc = cross_validated_auc(function, desc_ml)
    compare.append({"model": name, "mean AUC over the five held-out parts": mean_auc, "standard deviation": sd_auc, "seconds": time.time() - started})
show_table(pd.DataFrame(compare), floatfmt=".3f")

fit_rows, held = folds_ml[0]
curves = {}
for eta in (0.5, 0.05):
    booster = NewtonBooster(rounds=200, eta=eta, depth=3, lam=1.0).fit(desc_ml[fit_rows], y_ml[fit_rows], desc_ml[held], y_ml[held], watch_every=5)
    curves[eta] = pd.DataFrame(booster.history)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for axis, eta in zip(axes, curves):
    axis.plot(curves[eta]["round"], curves[eta]["AUC on the fitted compounds"], color=theme["blue"], lw=2, label="compounds the model was fitted on")
    axis.plot(curves[eta]["round"], curves[eta]["AUC on the watched compounds"], color=theme["red"], lw=2, label="held-out compounds")
    axis.set(xlabel="boosting rounds", title=f"learning rate eta = {eta}")
    tidy(axis, theme)
axes[0].set(ylabel="AUC")
axes[0].legend(frameon=False, loc="lower right")
fig.tight_layout()
booster_auc, lightgbm_auc = compare[0]["mean AUC over the five held-out parts"], compare[1]["mean AUC over the five held-out parts"]
best_round = {eta: int(curves[eta].loc[curves[eta]["AUC on the watched compounds"].idxmax(), "round"]) for eta in curves}
final_gap = {eta: float(curves[eta]["AUC on the fitted compounds"].iloc[-1] - curves[eta]["AUC on the watched compounds"].iloc[-1]) for eta in curves}
'''

PRESETS = '''
# The two LightGBM settings that the project's grid uses (nested_runners.LGBM_PRESETS), and what they do on feature set A in the five folds.
from cypmode.challenge.nested_runners import LGBM_PRESETS

preset_table = pd.DataFrame({name: pd.Series({k: f"{v:g}" for k, v in params.items()}) for name, params in LGBM_PRESETS.items()}).fillna("default").reset_index().rename(columns={"index": "setting"})
show_table(preset_table)
preset_rows = []
for name, params in LGBM_PRESETS.items():
    started = time.time()
    mean_auc, sd_auc = cross_validated_auc(lambda a, b, c, params=params: LGBMClassifier(verbose=-1, n_jobs=2, random_state=0, **params).fit(a, b).predict_proba(c)[:, 1], set_a)
    preset_rows.append({"setting": name, "mean AUC (5 folds)": mean_auc, "standard deviation": sd_auc, "seconds": time.time() - started})
show_table(pd.DataFrame(preset_rows), floatfmt=".3f")
'''

VARIANCE_OF_MEAN = '''
# The variance of an average of B correlated estimates, simulated and from the formula rho sigma^2 + (1 - rho) sigma^2 / B, for sigma = 1: B estimates with a common pairwise correlation rho.
rng = np.random.default_rng(3)
var_rows = []
for rho in (0.0, 0.3, 0.7):
    for B in (1, 10, 100):
        shared = rng.normal(size=(200_000, 1))
        own = rng.normal(size=(200_000, B))
        estimates = np.sqrt(rho) * shared + np.sqrt(1 - rho) * own                    # every pair has correlation rho and every estimate has variance 1
        var_rows.append({"correlation rho": rho, "trees B": B, "variance of the average, simulated": float(estimates.mean(axis=1).var()), "formula": rho + (1 - rho) / B})
show_table(pd.DataFrame(var_rows), floatfmt=".3f")
unique_share = {n: float(np.mean([len(np.unique(rng.integers(0, n, n))) / n for _ in range(200)])) for n in (100, 2334)}
print("share of the compounds that appear in a bootstrap sample of the same size:", {n: round(v, 3) for n, v in unique_share.items()}, f"; the limit 1 - 1/e = {1 - np.exp(-1):.3f}")
'''

FOREST = '''
# One tree, a forest of bootstrapped trees that may use every descriptor at a split (bagging), a random forest that may look at a tenth of the columns at each split, and extremely randomised trees, all on
# feature set A in the five folds; and how alike the trees of a forest are: the mean correlation between the predictions of two trees over the held-out compounds.
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

def forest(kind):
    if kind == "bagging":
        return RandomForestClassifier(n_estimators=150, max_features=1.0, min_samples_leaf=2, n_jobs=2, random_state=0)
    if kind == "forest":
        return RandomForestClassifier(n_estimators=150, max_features=0.1, min_samples_leaf=2, n_jobs=2, random_state=0)
    return ExtraTreesClassifier(n_estimators=150, max_features=0.1, min_samples_leaf=2, n_jobs=2, random_state=0)

forest_rows = []
single = cross_validated_auc(lambda a, b, c: DecisionTreeClassifier(min_samples_leaf=2, random_state=0).fit(a, b).predict_proba(c)[:, 1], desc_ml)
forest_rows.append({"model": "one decision tree (descriptors)", "mean AUC (5 folds)": single[0], "standard deviation": single[1], "mean correlation between two trees": float("nan")})
for kind, title in (("bagging", "150 bootstrapped trees, every column at every split"), ("forest", "random forest: 150 trees, a tenth of the columns at a split"), ("extra", "extremely randomised trees: the same with random thresholds")):
    fit_rows, held = folds_ml[0]
    model = forest(kind).fit(set_a[fit_rows], y_ml[fit_rows])
    tree_scores = np.array([tree.predict_proba(set_a[held])[:, 1] for tree in model.estimators_[:40]])
    correlation = np.corrcoef(tree_scores)
    mean_corr = float(correlation[np.triu_indices(40, 1)].mean())
    mean_auc, sd_auc = cross_validated_auc(lambda a, b, c, kind=kind: forest(kind).fit(a, b).predict_proba(c)[:, 1], set_a)
    forest_rows.append({"model": title, "mean AUC (5 folds)": mean_auc, "standard deviation": sd_auc, "mean correlation between two trees": mean_corr})
show_table(pd.DataFrame(forest_rows), floatfmt=".3f")
tree_auc = forest_rows[0]["mean AUC (5 folds)"]
bag_auc, forest_auc, extra_auc = forest_rows[1]["mean AUC (5 folds)"], forest_rows[2]["mean AUC (5 folds)"], forest_rows[3]["mean AUC (5 folds)"]
bag_corr, forest_corr, extra_corr = forest_rows[1]["mean correlation between two trees"], forest_rows[2]["mean correlation between two trees"], forest_rows[3]["mean correlation between two trees"]
'''

CELLS = [
    md("""
# Part IV. Machine learning from zero

Part III turned molecules into rows of numbers. This part explains what a computer does with such a table to learn a rule, and builds, from the equations up, every kind of model that the project uses or tested: trees (4.2), boosting (4.3), forests (4.4), similarity-based models (4.5),
regression for the two assay arms (4.6), neural networks (4.7), networks on graphs and on text (4.8), a table-reading pretrained model and principal components (4.9), and the numbers by which models are judged (4.10). Every model is first written out on the project's data
in a few lines, so that no part of it is a black box, and then compared with the library that the project actually runs. The data are those of a single task, CYP3A4 on the compounds with both arms; the results of the full nested evaluation, which is the honest one, are in Parts VI and VIII.

## 4.1 Supervised learning, loss functions and the problem of generalisation

> **Field:** Machine learning · Statistics · Mathematics

**The task.** We have $n$ examples. Each is a pair $(x_i, y_i)$: $x_i$ is a row of $p$ numbers describing compound $i$ (its **features**, section 3.7) and $y_i$ is the thing to be predicted, here 1 if the compound is a time-dependent inhibitor and 0 if not (its **label**). A **model** is a function $f_\\theta(x)$ with adjustable
numbers $\\theta$, its **parameters**; given a compound it produces a **score**. **Supervised learning** chooses $\\theta$ so that the scores agree with the labels of the examples it was shown, *and* so that they will also agree with the labels of compounds it was not shown. A **loss function** $L(y, f)$ measures how bad a score $f$ is for a compound with label $y$; the **empirical risk** is its average over the examples,
$R(\\theta) = \\frac1n\\sum_{i=1}^{n} L\\bigl(y_i, f_\\theta(x_i)\\bigr)$, and training means finding $\\theta$ that makes $R$ small, in most methods by **gradient descent**: repeatedly moving $\\theta$ a small step $\\eta$ (the **learning rate**) against the gradient, $\\theta \\leftarrow \\theta - \\eta\\,\\nabla_\\theta R$, because the gradient points to where $R$ grows fastest.

**The log-loss, from probability.** For a yes/no label it is natural to let the model say how likely the answer is yes. The score is the **logit** $z = f_\\theta(x)$, any real number, and the probability is $p = \\sigma(z)$ with the **sigmoid** $\\sigma(z) = 1/(1 + e^{-z})$, which maps the whole real line to the interval from 0 to 1 and is 0.5 at $z = 0$. If the model's probability is $p$, the probability that it
assigns to the observed label $y \\in \\{0, 1\\}$ is $p^{\\,y}(1 - p)^{1-y}$. For $n$ independent compounds the probability of all observed labels, the **likelihood**, is the product of these, and **maximum likelihood** chooses the parameters that make it largest. Maximising a product is the same as minimising the negative of its logarithm (the logarithm is increasing, and it turns the product into a sum), which gives the
**log-loss** (also called cross-entropy):

$$L(y, z) \\;=\\; -\\bigl[\\,y\\log p + (1 - y)\\log(1 - p)\\,\\bigr], \\qquad p = \\sigma(z).$$

Its derivative with respect to the logit is the quantity that every method below uses. With $d\\sigma/dz = \\sigma(1 - \\sigma) = p(1-p)$,

$$\\frac{\\partial L}{\\partial z} \\;=\\; -\\Bigl[\\frac{y}{p} - \\frac{1 - y}{1 - p}\\Bigr]\\,p(1 - p) \\;=\\; -\\bigl[y(1 - p) - (1 - y)p\\bigr] \\;=\\; p - y.$$

The gradient is simply the probability minus the label: a positive compound given $p = 0.9$ has gradient $-0.1$ (a little push upward), given $p = 0.1$ it has $-0.9$ (a strong push). The loss itself punishes confident mistakes severely, as the cell shows. For a number to be predicted, as the potency is, the usual loss is the **squared error** $L(y, f) = (y - f)^2$; the score that minimises its average over all compounds with the same features is the **conditional mean**
$\\mathbb{E}[y \\mid x]$, because $\\frac{d}{df}\\mathbb{E}[(y - f)^2 \\mid x] = -2\\,(\\mathbb{E}[y \\mid x] - f)$ vanishes at $f = \\mathbb{E}[y \\mid x]$. The cell checks the derivative of the log-loss by a finite difference.
"""),
    code(ML_DATA),
    code(LOGLOSS),
    md("""
The derivative $p - y$ agrees with the finite difference to within {{=worst_gap:.1e}}, and the loss for a positive compound grows from {{=penalty[0.9]:.2f}} at $p = 0.9$ to {{=penalty[0.01]:.1f}} at $p = 0.01$: a model that is sure and wrong pays far more than a model that hedges.

**Fitting is not the goal; predicting new compounds is.** A model with enough freedom can reproduce the labels of the training compounds exactly and still know nothing about new ones, as a student who memorised the answers of last year's exam. The error on *new* data has three parts, which the **bias-variance decomposition** separates. Suppose the truth is
$y = g(x) + \\varepsilon$, where $g$ is the true rule and the noise $\\varepsilon$ has mean 0 and variance $\\sigma^2$, and a model $\\hat f$ is fitted to a random training set (so that $\\hat f$ is itself random). The expected squared error at a point $x$, over both the training set and the noise of the new compound, is, after adding and subtracting $\\mathbb{E}[\\hat f(x)]$ and $g(x)$ and noting that the cross
terms vanish because the noise and the training set are independent and have mean zero,

$$\\mathbb{E}\\bigl[(y - \\hat f(x))^2\\bigr] \\;=\\; \\underbrace{\\bigl(g(x) - \\mathbb{E}[\\hat f(x)]\\bigr)^2}_{\\text{bias}^2} \\;+\\; \\underbrace{\\mathbb{E}\\bigl[(\\hat f(x) - \\mathbb{E}[\\hat f(x)])^2\\bigr]}_{\\text{variance}} \\;+\\; \\underbrace{\\sigma^2}_{\\text{noise}}.$$

**Bias** is the systematic error: how far the model's average prediction, over all possible training sets, is from the truth; a model that is too simple (a straight line for a curve) has high bias. **Variance** is how much the prediction changes when the training set changes; a model that is too flexible follows the noise of each particular training set and has high variance. The **noise** cannot be removed by any model. More flexibility lowers the bias and raises the variance, and
the best model is in between: the **bias-variance trade-off**. The cell measures all three on a case in which the truth is known.
"""),
    code(BIAS_VARIANCE),
    md("""
The error on the training points falls steadily with the degree of the polynomial, while the error on new points is lowest at degree {{=best_degree}} and rises again: low degrees are dominated by bias, high degrees by variance. The three parts add up to the measured error of new points to within {{=decomposition_gap:.3f}}, the sampling error of the simulation. The practical lesson is that the score of a model on the data it was fitted on is
not a measure of anything, and every number in this notebook that is meant as an estimate of performance is computed on compounds that the model did not see (Part VI).

## 4.2 Decision trees: impurity, and the search for the best question

> **Field:** Machine learning

A **decision tree** is a flowchart of yes/no questions about the features: "is the logP above 3.1?", then, on the "yes" branch, "is the molecular weight above 420?", and so on. A compound follows the answers from the top (the **root**) down to a **leaf**, and the leaf says what to predict: the fraction of positive training compounds that arrived there. A tree is built from the top, one question at a time, by the **greedy** rule: at every node pick the
question that makes the two resulting groups as pure as possible, and then do the same in each group.

**Impurity.** For a group in which a fraction $p$ of the compounds are positive, a measure of impurity should be 0 when $p = 0$ or $p = 1$ (a pure group) and largest at $p = 1/2$. Two are used. The **Gini impurity** is the probability that two compounds drawn at random from the group (with replacement) belong to different classes, which for two classes is
$G = 2p(1 - p) = 1 - p^2 - (1-p)^2$; the **entropy** is $H = -p\\log_2 p - (1-p)\\log_2(1-p)$, the number of bits of information needed to say which class a randomly drawn compound belongs to. They behave alike, and the project's trees use the Gini impurity.

**The split search.** A question is a feature $j$ and a threshold $t$: the compounds with $x_j \\le t$ go left, the others right. With $n_L$ and $n_R$ compounds in the two parts ($n = n_L + n_R$), the quality of the question is the **impurity decrease** $\\Delta$

$$\\Delta \\;=\\; G(\\text{parent}) \\;-\\; \\frac{n_L}{n}\\,G(\\text{left}) \\;-\\; \\frac{n_R}{n}\\,G(\\text{right}),$$

the impurity of the group before the split minus the average impurity after it, weighted by size. Only the thresholds between two consecutive different values of a feature need to be tried, and after sorting the compounds by that feature the counts of positives on the left of every position are cumulative sums, so all thresholds of one feature cost one pass. The cell does this for all {{=len(descriptor_names)}} descriptors and compares it with
scikit-learn's tree of depth 1.
"""),
    code(STUMP),
    md("""
The search chooses the descriptor **{{=stump_rows[0]['descriptor']}}** at the threshold {{=stump_rows[0]['threshold']:.3f}}, with an impurity decrease of {{=stump_rows[0]['impurity decrease']:.4f}}, and the library makes the same choice ({{=same_split}}). Below the threshold {{=stump_rows[0]['share positive, below']:.0%}} of the compounds are positive, above it {{=stump_rows[0]['share positive, above']:.0%}}, against {{=positives_ml / n_ml:.0%}} in the whole set.

**What a tree does when it is allowed to grow.** With one question the tree is very simple (high bias). Every further level can separate the training compounds better, and with no limit a tree separates them perfectly, a leaf for each compound or small group, which is variance at its worst. The next cell shows both effects and a third: the first question of a tree is not stable. Trees fitted to slightly different samples of the data choose different roots.
"""),
    code(TREE_BEHAVIOUR),
    md("""
A tree without a limit reaches an AUC of {{=fitted_unlimited:.3f}} on the compounds it was fitted on and {{=held_unlimited:.3f}} on held-out ones: the first number is the memorising student, the second the real one. The best depth among those tried is {{=best_depth_row['depth of the tree']}}, with {{=best_depth_row['AUC on held-out compounds']:.3f}}. The root of a tree of depth 2 is one of {{=n_distinct_roots}} descriptors over 60 **bootstrap samples** (each is drawn with replacement from the compounds, as section 4.4 explains), the most frequent in {{=root_top_share:.0%}} of them, and the three questions at the top of the tree take {{=n_shapes}} different combinations, the most frequent in only {{=shape_top_share:.0%}} of the trees: a
single tree is an unstable model, the more so the deeper it is. Both faults, the memorising and the instability, have one remedy that all the strong models below use: do not trust one tree but combine many. They differ in how the trees are made and combined. The figure shows a tree fitted to real compounds, with the question and the composition of each node.
"""),
    figure("ml_decision_tree"),
    md("""
## 4.3 Gradient boosting: many small trees, each correcting the others

> **Field:** Machine learning · Mathematics (optimisation)

**The idea.** Build the model in stages, each stage a small tree that repairs what the previous stages still get wrong. Start with a constant (the prior log-odds of the positive class) and let $F_{m-1}(x)$ be the logit after $m - 1$ trees; the new tree $f_m$ is added with a small weight, $F_m = F_{m-1} + \\eta\\,f_m$, where $\\eta$ is the **learning rate**, usually 0.01 to 0.3: a tree is only allowed to correct a part of the error, so that no
single tree can dominate. Friedman (2001) saw that this is gradient descent, not on parameters but on the function itself. The loss of the whole data as a function of the $n$ scores $F(x_1), \\dots, F(x_n)$ has the gradient $g_i = \\partial L(y_i, F)/\\partial F$ at $F_{m-1}$, and the best small step downhill is to move every score by a multiple of $-g_i$; since a step that is defined only at the training compounds is useless for new ones, a tree is fitted to $-g_i$ and
so generalises the step to every compound. For the log-loss $-g_i = y_i - p_i$ is the residual of the probability, the part of the label that is not yet explained.

**The second-order version used by LightGBM.** XGBoost (Chen and Guestrin, 2016) and LightGBM (Ke and co-workers, 2017) do not fit the residual by least squares; they take, as Newton's method does, the curvature into account. At round $m$ the objective for a new tree $f$ with $T$ leaves and leaf values $w_1, \\dots, w_T$ is the loss of the data plus a penalty for complexity,
$\\sum_i L\\bigl(y_i, F_{m-1}(x_i) + f(x_i)\\bigr) + \\gamma T + \\tfrac12 \\lambda \\sum_j w_j^2$, where $\\gamma$ and $\\lambda$ are non-negative constants that punish trees with many leaves and large leaf values. The loss is approximated by the first terms of its Taylor series in $f$,
$L(y_i, F + f) \\approx L(y_i, F) + g_i f + \\tfrac12 h_i f^2$, where $g_i = p_i - y_i$ is the gradient and $h_i = \\partial^2 L/\\partial F^2 = p_i(1 - p_i)$ the curvature of the log-loss (the derivative of $p - y$ with respect to $z$ is $p(1-p)$). A tree assigns every compound to one leaf $j$ and gives it the value $w_j$ of that leaf. Collect the compounds of leaf $j$ in a set $I_j$ and write
$G_j = \\sum_{i \\in I_j} g_i$ and $H_j = \\sum_{i \\in I_j} h_i$; dropping the terms that do not depend on the tree, the objective becomes a sum over leaves,

$$\\sum_{j=1}^{T}\\Bigl[\\,G_j w_j + \\tfrac12 (H_j + \\lambda)\\,w_j^2\\Bigr] + \\gamma T .$$

Each bracket is a parabola in $w_j$. Setting its derivative to zero gives the best **leaf value** and the value of the objective that it reaches,

$$w_j^{*} = -\\frac{G_j}{H_j + \\lambda}, \\qquad \\text{objective}^{*} = -\\frac12\\sum_{j=1}^{T}\\frac{G_j^2}{H_j + \\lambda} + \\gamma T .$$

This is a score for a whole tree, so splitting a leaf into a left part (index $L$) and a right part ($R$) can be judged by how much it lowers the objective, the **gain**,

$$\\text{gain} \\;=\\; \\frac12\\Bigl[\\frac{G_L^2}{H_L + \\lambda} + \\frac{G_R^2}{H_R + \\lambda} - \\frac{(G_L + G_R)^2}{H_L + H_R + \\lambda}\\Bigr] - \\gamma ,$$

and a split is made only if the gain is positive. The **split search** is then exactly that of section 4.2 with a different quality measure, computed from sums of $g$ and $h$ that can be accumulated in one pass. LightGBM makes this fast with two ideas. It **bins** every feature into at most 255 values once, so that the sums of $g$ and $h$ are accumulated per bin (a *histogram*) and the best threshold is found among the bin edges instead of among all values, and it grows the tree **leaf-wise**: instead of splitting all nodes of one level, it always splits the leaf with the largest gain, until the allowed number of leaves is
reached, which gives a lopsided tree that reduces the loss faster than a balanced one. The following cell writes the whole algorithm out, in about forty lines, with the equations above, on the descriptors of the 2334 compounds.
"""),
    code(NEWTON),
    md("""
The formula predicts a decrease of {{=predicted_decrease:.2f}} in the summed log-loss for the best single split, and the true decrease is {{=actual_decrease:.2f}}; the two differ because the formula rests on a second-order approximation of the loss, which is good for a small step and not exact for a large one. Next the written-out booster is compared with LightGBM at matching settings in the same five folds, and the effect of the learning rate is shown.
"""),
    code(NEWTON_CV),
    figure("ml_boosting_stages"),
    md("""
The written-out booster reaches a mean AUC of {{=booster_auc:.3f}} against {{=lightgbm_auc:.3f}} for LightGBM with the same settings, which differ in details that the forty lines leave out (the way LightGBM grows leaf-wise and bins its features, and its handling of small leaves); the two agree to within the spread between folds. On the first fold the held-out AUC is highest after {{=best_round[0.5]}} rounds for $\\eta = 0.5$ and after {{=best_round[0.05]}} for $\\eta = 0.05$,
and by round 200 the gap between the AUC on the fitted and on the held-out compounds is {{=final_gap[0.5]:.2f}} and {{=final_gap[0.05]:.2f}}: a large learning rate fits the training data quickly and begins to memorise it sooner, a small one needs more rounds and fits more gently. The number of rounds and the learning rate are therefore chosen together. The figure above follows a boosted model stage by stage on a real task.

**The settings of the project.** The project's candidates use LightGBM with two presets (`LGBM_PRESETS` in `cypmode/challenge/nested_runners.py`): a **deep** one (300 trees, learning rate 0.04, up to 31 leaves per tree) and a **shallow** one (600 trees, learning rate 0.02, up to 8 leaves, and an $L_2$ penalty $\\lambda = 5$ on the leaf values), each with **row subsampling** (every tree sees 70 to 80 % of the compounds, drawn without replacement every round) and **column subsampling** (every tree sees 20 to 30 % of the columns), and a minimum number of compounds in a leaf. Subsampling is another
source of randomness that decorrelates the trees (section 4.4) and speeds up the fit. The cell prints the presets and compares them on feature set A.
"""),
    code(PRESETS),
    md("""
## 4.4 Random forests and extremely randomised trees: averaging many trees

> **Field:** Machine learning · Statistics

Boosting builds trees in sequence, each depending on the earlier ones. A different way to use many trees is to build them independently and average their scores. Averaging helps because it reduces variance. Suppose $B$ estimates $X_1, \\dots, X_B$ (written $X_b$, with $b$ running from 1 to $B$) of the same quantity each have variance $\\sigma^2$, and every pair has the same correlation $\\rho$. The variance of their average is

$$\\mathrm{Var}\\Bigl(\\frac1B\\sum_{b=1}^{B} X_b\\Bigr) \\;=\\; \\frac{1}{B^2}\\Bigl[B\\sigma^2 + B(B-1)\\rho\\sigma^2\\Bigr] \\;=\\; \\rho\\,\\sigma^2 + \\frac{1 - \\rho}{B}\\,\\sigma^2 ,$$

where the sum of variances gives $B\\sigma^2$ and the $B(B-1)$ ordered pairs give the covariances $\\rho\\sigma^2$. When the estimates are independent ($\\rho = 0$) the variance falls as $1/B$; when they are correlated, adding more of them stops helping beyond the floor $\\rho\\sigma^2$. So the trees should be **as different from one another as possible**, while each stays a good model. **Bagging** (bootstrap aggregation, Breiman, 1996) makes them differ by training each tree on a **bootstrap sample**:
$n$ compounds drawn with replacement from the $n$ available, so that a given compound is missing from a sample with probability $(1 - 1/n)^n \\to e^{-1} \\approx 0.37$, and every sample holds about 63 % of the distinct compounds (the compounds left out, the **out-of-bag** ones, give a free estimate of the error). The **random forest** (Breiman, 2001) adds a second source of difference: at every split a tree may consider only a random subset of the
columns, so that one dominant feature cannot make the first question of every tree the same. **Extremely randomised trees** (Geurts, Ernst and Wehenkel, 2006) go further: the threshold of every candidate column is drawn at random instead of being searched, and the best of those random questions is used, which makes the trees even less alike and the fitting faster, at the price of a little more bias. The cell checks the variance formula by simulation and the share of compounds in a bootstrap sample.
"""),
    code(VARIANCE_OF_MEAN),
    md("""
The simulated variances agree with the formula: independent trees ($\\rho = 0$) average away the variance, and for $\\rho = 0.7$ the variance cannot go below 0.7 however many trees are added. Only {{=unique_share[2334] * 100:.1f}} % of the compounds appear in a bootstrap sample of 2334, the limit being $1 - e^{-1} = 63.2$ %. The next cell compares one tree with bagged trees, a random forest and extremely randomised trees on feature set A, and measures how alike the trees of each forest are.
"""),
    code(FOREST),
    md("""
One tree reaches {{=tree_auc:.3f}}; bagging 150 of them gives {{=bag_auc:.3f}}, the random forest {{=forest_auc:.3f}} and the extremely randomised trees {{=extra_auc:.3f}}. The mean correlation between the predictions of two trees is {{=bag_corr:.2f}} when every column may be used at every split and {{=forest_corr:.2f}} with a tenth of the columns: looking at fewer columns does make the trees less alike, as the
formula asks. Random thresholds do not lower it further ({{=extra_corr:.2f}}), so whatever advantage the extremely randomised trees have on these data is not an advantage of lower correlation between trees; the project keeps both kinds as candidates and lets the validation decide. The first row of the table is a single tree on the descriptors only, as in section 4.2. The project's candidates `rf_A` and `extratrees_A` are of this kind (400 trees, a tenth of the columns at a split, at least two compounds in a leaf).
"""),
]
