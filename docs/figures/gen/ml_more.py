# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Concept figures of machine learning drawn from real compounds: bias and variance, bagging against boosting, the Tanimoto kernel, a small neural network trained by
backpropagation, message passing on a molecular graph, self-attention, principal components, and the confusion matrix with the Matthews correlation coefficient."""

import re
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import figure
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdDepictor
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from style import apply, tidy

from gen._stack import evaluation

GROUP = "ml"
ROOT = Path(__file__).resolve().parents[3]
TRAIN = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"
FINAL = "data/tdi_track/outputs/final/manifest.json"
DESCRIPTORS = ("MolLogP", "TPSA", "MolWt", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "RingCount", "FractionCSP3")
FLUCONAZOLE = "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F"  # the ligand of 6MA7, used wherever a figure needs one small real molecule


@cache
def labelled() -> pd.DataFrame:
    """The CYP3A4 compounds with both assay arms and a label, with eight descriptors and a Morgan fingerprint each."""
    train = pd.read_csv(ROOT / TRAIN)
    keep = train["CYP3A4_is_TDI"].notna() & train["CYP3A4_pIC50_TDI_condition"].notna() & train["CYP3A4_pIC50_direct_inhibition"].notna()
    table = train.loc[keep, ["SMILES", "CYP3A4_is_TDI"]].reset_index(drop=True)
    mols = [Chem.MolFromSmiles(s) for s in table["SMILES"]]
    for name in DESCRIPTORS:
        table[name] = [getattr(Descriptors, name)(m) for m in mols]
    table["label"] = table["CYP3A4_is_TDI"].astype(int)
    table["fp"] = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
    return table


def matrix(columns=DESCRIPTORS) -> tuple[np.ndarray, np.ndarray]:
    table = labelled()
    return table[list(columns)].to_numpy(float), table["label"].to_numpy()


def folds():
    return StratifiedKFold(5, shuffle=True, random_state=0)


# -- bias and variance: depth of a tree ----------------------------------------------------------------------------------------------------------

DEPTHS = tuple(range(1, 13))
BOOTSTRAPS = 20


@cache
def depth_scan() -> pd.DataFrame:
    x, y = matrix()
    rows = []
    for depth in DEPTHS:
        train_auc, cv_auc, spread = [], [], []
        for fit, held in folds().split(x, y):
            tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(x[fit], y[fit])
            train_auc.append(roc_auc_score(y[fit], tree.predict_proba(x[fit])[:, 1]))
            cv_auc.append(roc_auc_score(y[held], tree.predict_proba(x[held])[:, 1]))
        fit, held = next(iter(folds().split(x, y)))
        rng = np.random.default_rng(0)
        predictions = []
        for _ in range(BOOTSTRAPS):
            sample = rng.choice(fit, size=len(fit), replace=True)
            predictions.append(DecisionTreeClassifier(max_depth=depth, random_state=0).fit(x[sample], y[sample]).predict_proba(x[held])[:, 1])
        spread = float(np.mean(np.std(predictions, axis=0)))
        rows.append({"depth": depth, "train": np.mean(train_auc), "cv": np.mean(cv_auc), "cv_sd": np.std(cv_auc, ddof=1), "spread": spread})
    return pd.DataFrame(rows)


def bias_variance_caption() -> str:
    scan = depth_scan()
    best = scan.loc[scan["cv"].idxmax()]
    return (f"A decision tree of growing depth on {len(labelled())} real CYP3A4 compounds described by eight descriptors. Left: the AUC on the compounds the tree was fitted to (it keeps rising towards "
            f"{scan['train'].iloc[-1]:.2f}) and the five-fold cross-validated AUC (it peaks at depth {int(best['depth'])} with {best['cv']:.3f} and then falls to {scan['cv'].iloc[-1]:.3f}): the gap between the two is the optimism of a model that has memorised its data. "
            f"Right: the variance, measured as the mean standard deviation of the predictions of {BOOTSTRAPS} trees fitted to bootstrap resamples of the same training fold, for the same held-out compounds; it grows from {scan['spread'].iloc[0]:.3f} at depth 1 to "
            f"{scan['spread'].iloc[-1]:.3f} at depth {DEPTHS[-1]}. A shallow tree is stable and too simple (bias); a deep tree fits the training data and changes with every sample (variance). "
            "Forests average the variance away and boosting reduces the bias one small tree at a time, which is what the next two figures show.")


@figure("ml_bias_variance", GROUP, "Bias and variance: the depth of a tree on real compounds", bias_variance_caption, data=(TRAIN,), section="4.1")
def bias_variance(theme_name):
    th = apply(theme_name)
    scan = depth_scan()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.plot(scan["depth"], scan["train"], "o-", color=th["red"], label="on the data it was fitted to")
    ax.plot(scan["depth"], scan["cv"], "o-", color=th["blue"], label="held out (5-fold)")
    ax.fill_between(scan["depth"], scan["cv"] - scan["cv_sd"], scan["cv"] + scan["cv_sd"], color=th["blue"], alpha=0.18)
    ax.set(xlabel="depth of the tree", ylabel="AUC", title="what the tree reports and what a held-out fold shows")
    ax.legend(frameon=False)
    tidy(ax, th)
    ax = axes[1]
    ax.plot(scan["depth"], scan["spread"], "o-", color=th["orange"])
    ax.set(xlabel="depth of the tree", ylabel="spread of the prediction over bootstrap trees", title="variance: how much the answer depends on the sample")
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- bagging against boosting ---------------------------------------------------------------------------------------------------------------------

TREE_COUNTS = (1, 2, 5, 10, 20, 50, 100, 200)


@cache
def ensemble_scan() -> pd.DataFrame:
    x, y = matrix()
    rows = []
    for n in TREE_COUNTS:
        for name, model in (("bagging", lambda n=n: RandomForestClassifier(n_estimators=n, min_samples_leaf=5, random_state=0, n_jobs=2)),
                            ("boosting", lambda n=n: GradientBoostingClassifier(n_estimators=n, max_depth=3, learning_rate=0.1, random_state=0))):
            aucs = []
            for fit, held in folds().split(x, y):
                aucs.append(roc_auc_score(y[held], model().fit(x[fit], y[fit]).predict_proba(x[held])[:, 1]))
            rows.append({"method": name, "trees": n, "auc": np.mean(aucs), "sd": np.std(aucs, ddof=1)})
    return pd.DataFrame(rows)


@cache
def bagged_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Predictions of 100 bootstrap trees (each fitted to a resample of the first training fold) for 25 held-out compounds, and their mean."""
    x, y = matrix()
    fit, held = next(iter(folds().split(x, y)))
    held = held[:25]
    rng = np.random.default_rng(1)
    each = []
    for _ in range(100):
        sample = rng.choice(fit, size=len(fit), replace=True)
        each.append(DecisionTreeClassifier(min_samples_leaf=5, random_state=0).fit(x[sample], y[sample]).predict_proba(x[held])[:, 1])
    each = np.array(each)
    return each, each.mean(axis=0)


def ensemble_caption() -> str:
    scan = ensemble_scan()
    one = scan[scan["trees"] == 1].set_index("method")["auc"]
    last = scan[scan["trees"] == TREE_COUNTS[-1]].set_index("method")["auc"]
    each, _mean = bagged_predictions()
    return (f"Two ways of combining many trees, on the same {len(labelled())} compounds and eight descriptors. Left: five-fold cross-validated AUC against the number of trees. Bagging (a random forest: each tree sees a bootstrap resample, the trees are averaged) "
            f"starts from one noisy deep tree ({one['bagging']:.3f}) and climbs to {last['bagging']:.3f} with {TREE_COUNTS[-1]} trees; boosting (each shallow tree is fitted to what the trees before it got wrong) starts from {one['boosting']:.3f} and reaches {last['boosting']:.3f}. "
            f"Right: why averaging helps: each pale dot is the prediction of one of 100 bootstrap trees for one of 25 held-out compounds (the spread of a single tree is large, mean standard deviation {each.std(axis=0).mean():.2f}); the bold dot is the average, which has no such spread. "
            "Bagging reduces variance and works with deep trees; boosting reduces bias and works with shallow ones.")


@figure("ml_bagging_boosting", GROUP, "Bagging against boosting on real compounds", ensemble_caption, data=(TRAIN,), section="4.4")
def bagging_boosting(theme_name):
    th = apply(theme_name)
    scan = ensemble_scan()
    each, mean = bagged_predictions()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    for method, color in (("bagging", th["blue"]), ("boosting", th["red"])):
        part = scan[scan["method"] == method]
        ax.plot(part["trees"], part["auc"], "o-", color=color, label=method)
        ax.fill_between(part["trees"], part["auc"] - part["sd"], part["auc"] + part["sd"], color=color, alpha=0.15)
    ax.set_xscale("log")
    ax.set(xlabel="number of trees", ylabel="held-out AUC (5-fold)", title="more trees: two ways")
    ax.legend(frameon=False)
    tidy(ax, th)
    ax = axes[1]
    order = np.argsort(mean)
    for position, compound in enumerate(order):
        ax.scatter(np.full(each.shape[0], position), each[:, compound], s=6, color=th["muted"], alpha=0.35, linewidths=0)
        ax.scatter(position, mean[compound], s=34, color=th["blue"], zorder=3)
    ax.set(xlabel="25 held-out compounds, ordered by the averaged prediction", ylabel="predicted probability", title="bagging: 100 noisy trees, one steady average")
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- the Tanimoto kernel --------------------------------------------------------------------------------------------------------------------------

def tanimoto_matrix(fps_a, fps_b=None) -> np.ndarray:
    fps_b = fps_a if fps_b is None else fps_b
    return np.array([DataStructs.BulkTanimotoSimilarity(a, list(fps_b)) for a in fps_a])


@cache
def kernel_study() -> dict:
    table = labelled()
    rng = np.random.default_rng(0)
    shown = []
    for seed in rng.choice(len(table), 6, replace=False):  # six compounds and their seven nearest neighbours: six chemical series
        similarity = np.array(DataStructs.BulkTanimotoSimilarity(table["fp"][seed], list(table["fp"])))
        shown += [int(i) for i in np.argsort(-similarity)[:8] if int(i) not in shown]
    shown = np.array(sorted(set(shown)))
    k_shown = tanimoto_matrix([table["fp"][i] for i in shown])
    order = leaves_list(linkage(squareform(1 - k_shown, checks=False), method="average"))
    spectrum_rows = np.sort(rng.choice(len(table), 400, replace=False))
    eigenvalues = np.linalg.eigvalsh(tanimoto_matrix([table["fp"][i] for i in spectrum_rows]))[::-1]
    svm_rows = np.sort(rng.choice(len(table), 1200, replace=False))
    fps = [table["fp"][i] for i in svm_rows]
    k_svm, y = tanimoto_matrix(fps), table["label"].to_numpy()[svm_rows]
    scores = np.zeros(len(svm_rows))
    for fit, held in folds().split(k_svm, y):
        model = SVC(kernel="precomputed", C=1.0).fit(k_svm[np.ix_(fit, fit)], y[fit])
        scores[held] = model.decision_function(k_svm[np.ix_(held, fit)])
    return {"shown": shown[order], "k": k_shown[np.ix_(order, order)], "labels": table["label"].to_numpy()[shown][order], "eigenvalues": eigenvalues, "svm_auc": roc_auc_score(y, scores), "svm_n": len(svm_rows)}


def kernel_caption() -> str:
    study = kernel_study()
    return (f"The Tanimoto similarity of Morgan fingerprints used as the kernel of a support vector machine. Left: the kernel matrix of {len(study['labels'])} real CYP3A4 compounds (six compounds with their seven nearest neighbours each), ordered so that similar compounds sit together: the bright blocks on the diagonal are chemical series and the dark remainder is how little unrelated compounds share; "
            "the strip beside the matrix marks time-dependent inhibitors in red. Right: the eigenvalues of the kernel matrix of 400 other compounds; the smallest is "
            f"{study['eigenvalues'][-1]:.2e}, so no eigenvalue is negative beyond rounding: the Tanimoto similarity is a valid positive semi-definite kernel, which is what a support vector machine needs. "
            f"A support vector machine on this precomputed kernel, fitted and scored by five-fold cross-validation on {study['svm_n']} compounds, reaches an AUC of {study['svm_auc']:.3f}.")


@figure("ml_tanimoto_kernel", GROUP, "The Tanimoto similarity as a kernel", kernel_caption, data=(TRAIN,), section="4.5")
def tanimoto_kernel(theme_name):
    th = apply(theme_name)
    study = kernel_study()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list("k", [th["background"], th["blue"]])
    ax.imshow(study["k"], cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for position, label in enumerate(study["labels"]):
        ax.add_patch(plt.Rectangle((-3.2, position - 0.5), 1.8, 1, color=th["red"] if label else th["grid"], clip_on=False))
    ax.set_title(f"kernel matrix K(a, b) of {len(study['labels'])} compounds in six series")
    ax = axes[1]
    ax.plot(np.arange(1, len(study["eigenvalues"]) + 1), study["eigenvalues"], color=th["orange"], lw=1.6)
    ax.axhline(0, color=th["text"], lw=1)
    ax.set(xlabel="rank of the eigenvalue", ylabel="eigenvalue of the kernel matrix", title="no negative eigenvalue: a valid kernel", yscale="symlog", yticks=[0, 1, 10, 100], ylim=(-0.3, 120))
    ax.set_yticklabels(["0", "1", "10", "100"])
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- a neural network and backpropagation ---------------------------------------------------------------------------------------------------------

HIDDEN, EPOCHS, RATE = 6, 400, 0.5


def forward(params, x):
    w1, b1, w2, b2 = params
    h = np.tanh(x @ w1 + b1)
    p = 1.0 / (1.0 + np.exp(-(h @ w2 + b2)))
    return h, p


def loss_of(params, x, y):
    p = forward(params, x)[1].ravel()
    return float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))


def gradients(params, x, y):
    """Backpropagation by hand: the error at the output, pushed back through the weights and the derivative of tanh."""
    _w1, _b1, w2, _b2 = params
    h, p = forward(params, x)
    d_out = (p.ravel() - y)[:, None] / len(y)
    g_w2, g_b2 = h.T @ d_out, d_out.sum(0)
    d_hidden = (d_out @ w2.T) * (1 - h**2)
    return [x.T @ d_hidden, d_hidden.sum(0), g_w2, g_b2]


@cache
def trained_network() -> dict:
    x_raw, y = matrix(("MolLogP", "TPSA"))
    scaler = StandardScaler().fit(x_raw)
    x = scaler.transform(x_raw)
    rng = np.random.default_rng(0)
    params = [rng.normal(0, 0.7, (2, HIDDEN)), np.zeros(HIDDEN), rng.normal(0, 0.7, (HIDDEN, 1)), np.zeros(1)]
    check = [rng.normal(0, 0.5, p.shape) for p in params]
    analytic = gradients(check, x[:200], y[:200])
    worst = 0.0
    for index, (p, g) in enumerate(zip(check, analytic, strict=True)):
        for position in np.ndindex(p.shape):
            shifted = [q.copy() for q in check]
            shifted[index][position] += 1e-5
            up = loss_of(shifted, x[:200], y[:200])
            shifted[index][position] -= 2e-5
            numeric = (up - loss_of(shifted, x[:200], y[:200])) / 2e-5
            worst = max(worst, abs(numeric - g[position]) / (abs(numeric) + abs(g[position]) + 1e-12))
    fit, held = next(iter(folds().split(x, y)))
    losses = []
    for _ in range(EPOCHS):
        losses.append(loss_of(params, x[fit], y[fit]))
        for p, g in zip(params, gradients(params, x[fit], y[fit]), strict=True):
            p -= RATE * g
    auc_held = roc_auc_score(y[held], forward(params, x[held])[1].ravel())
    return {"params": params, "losses": losses, "x": x, "y": y, "scaler": scaler, "worst_gradient_error": worst, "auc_held": auc_held, "epochs": EPOCHS}


def backprop_caption() -> str:
    net = trained_network()
    return (f"A neural network with two inputs (MolLogP and TPSA, standardised), {HIDDEN} hidden units with tanh and one sigmoid output, trained on real CYP3A4 compounds by gradient descent for {net['epochs']} steps; the code is written out in the figure's module with the backward pass by hand. "
            f"Left: the network, each line drawn thicker where its trained weight is larger (blue positive, red negative). Middle: the cross-entropy loss falls from {net['losses'][0]:.3f} to {net['losses'][-1]:.3f} on the training fold. "
            f"Right: the probability of time-dependent inhibition over the plane of the two descriptors, with the compounds; the AUC on the held-out fold is {net['auc_held']:.3f}, close to what the depth-3 tree of figure 4.2 reached, because two descriptors carry little more. "
            f"The backward pass is checked against finite differences: the largest relative difference between the gradient of backpropagation and the numerical gradient of every one of the {sum(p.size for p in net['params'])} parameters is {net['worst_gradient_error']:.1e}.")


@figure("ml_backprop", GROUP, "A small neural network trained by backpropagation", backprop_caption, data=(TRAIN,), section="4.7")
def backprop(theme_name):
    th = apply(theme_name)
    net = trained_network()
    w1, _b1, w2, _b2 = net["params"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.1, 1, 1.2]})
    ax = axes[0]
    ax.axis("off")
    ys_in, ys_hidden = [0.35, 0.65], np.linspace(0.12, 0.88, HIDDEN)
    scale = max(np.abs(w1).max(), np.abs(w2).max())
    for i, yi in enumerate(ys_in):
        for j, yj in enumerate(ys_hidden):
            ax.plot([0.12, 0.5], [yi, yj], color=th["blue"] if w1[i, j] > 0 else th["red"], lw=0.4 + 3.4 * abs(w1[i, j]) / scale, alpha=0.8)
    for j, yj in enumerate(ys_hidden):
        ax.plot([0.5, 0.88], [yj, 0.5], color=th["blue"] if w2[j, 0] > 0 else th["red"], lw=0.4 + 3.4 * abs(w2[j, 0]) / scale, alpha=0.8)
    ax.scatter([0.12] * 2, ys_in, s=380, color=th["panel"], edgecolor=th["text"], zorder=3)
    ax.scatter([0.5] * HIDDEN, ys_hidden, s=300, color=th["panel"], edgecolor=th["text"], zorder=3)
    ax.scatter([0.88], [0.5], s=380, color=th["panel"], edgecolor=th["text"], zorder=3)
    ax.text(0.12, 0.2, "MolLogP\nTPSA", ha="center", fontsize=8, color=th["muted"], va="top")
    ax.text(0.5, 0.0, "hidden (tanh)", ha="center", fontsize=8, color=th["muted"])
    ax.text(0.88, 0.4, "P(TDI)", ha="center", fontsize=8, color=th["muted"], va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1)
    ax.set_title("the network and its trained weights")
    ax = axes[1]
    ax.plot(net["losses"], color=th["orange"], lw=2)
    ax.set(xlabel="gradient step", ylabel="cross-entropy loss", title="backpropagation lowers the loss")
    tidy(ax, th)
    ax = axes[2]
    x_raw, y = matrix(("MolLogP", "TPSA"))
    gx, gy = np.meshgrid(np.linspace(np.percentile(x_raw[:, 0], 1), np.percentile(x_raw[:, 0], 99), 120), np.linspace(np.percentile(x_raw[:, 1], 1), np.percentile(x_raw[:, 1], 99), 120))
    grid = net["scaler"].transform(np.c_[gx.ravel(), gy.ravel()])
    probability = forward(net["params"], grid)[1].reshape(gx.shape)
    cmap = LinearSegmentedColormap.from_list("p", [th["blue"], th["background"], th["red"]])
    image = ax.contourf(gx, gy, probability, levels=12, cmap=cmap, alpha=0.85)
    ax.scatter(x_raw[::6, 0], x_raw[::6, 1], c=np.where(y[::6] == 1, th["red"], th["blue"]), s=6, alpha=0.5, linewidths=0)
    fig.colorbar(image, ax=ax, label="P(time-dependent inhibitor)")
    ax.set(xlabel="MolLogP", ylabel="TPSA", title="what it learned", xlim=(gx.min(), gx.max()), ylim=(gy.min(), gy.max()))
    fig.tight_layout()
    return fig


# -- message passing on a molecular graph --------------------------------------------------------------------------------------------------------

def drawn_molecule(smiles: str = FLUCONAZOLE):
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    coords = np.array([list(mol.GetConformer().GetAtomPosition(i))[:2] for i in range(mol.GetNumAtoms())])
    return mol, coords, Chem.GetDistanceMatrix(mol)


ELEMENT_COLORS = {"C": "muted", "N": "blue", "O": "red", "F": "green"}


def message_passing_caption() -> str:
    mol, _coords, distances = drawn_molecule()
    start = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetIsAromatic() and a.GetDegree() == 2)
    reached = [int((distances[start] <= k).sum()) for k in range(4)]
    return (f"Message passing on the graph of fluconazole ({mol.GetNumAtoms()} heavy atoms). In each round every atom collects the messages of its bonded neighbours and updates its own vector; after k rounds an atom's vector depends on exactly the atoms within k bonds. "
            f"The panels follow one atom, the ring nitrogen that binds the iron of the enzyme (ringed): after 0 rounds it knows only itself, after 1 round {reached[1]} atoms, after 2 rounds {reached[2]} and after 3 rounds {reached[3]} of the {mol.GetNumAtoms()} atoms (shaded). "
            "A graph network of depth three therefore describes each atom by its three-bond environment, the learned counterpart of the circular Morgan environments of section 3.1, and the vector of the whole molecule is a pooled sum of the atom vectors.")


@figure("ml_message_passing", GROUP, "Message passing: what an atom knows after k rounds", message_passing_caption, data=("the structure of fluconazole (the ligand of 6MA7)",), section="4.8")
def message_passing(theme_name):
    th = apply(theme_name)
    mol, coords, distances = drawn_molecule()
    start = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetIsAromatic() and a.GetDegree() == 2)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for k, ax in enumerate(axes):
        ax.axis("off")
        ax.set_aspect("equal")
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            inside = distances[start][a] <= k and distances[start][b] <= k
            ax.plot(coords[[a, b], 0], coords[[a, b], 1], color=th["blue"] if inside else th["grid"], lw=2.6 if inside else 1.4, zorder=1)
        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            inside = distances[start][i] <= k
            ax.scatter(*coords[i], s=190 if i != start else 300, color=th["panel"] if not inside else th["blue"], edgecolor=th[ELEMENT_COLORS.get(atom.GetSymbol(), "muted")] if i != start else th["gold"], linewidth=2.2 if i != start else 3.2, zorder=2)
            if atom.GetSymbol() != "C":
                ax.text(*coords[i], atom.GetSymbol(), ha="center", va="center", fontsize=7.5, color=th["text"] if not inside else th["background"], zorder=3)
        ax.set_xlim(coords[:, 0].min() - 0.8, coords[:, 0].max() + 0.8)
        ax.set_ylim(coords[:, 1].min() - 0.8, coords[:, 1].max() + 0.8)
        ax.set_title(f"after {k} round{'s' if k != 1 else ''}: {int((distances[start] <= k).sum())} atoms", fontsize=10)
    fig.tight_layout()
    return fig


# -- self-attention -------------------------------------------------------------------------------------------------------------------------------

TOKEN = re.compile(r"(\[[^\]]+]|Br|Cl|N|O|S|P|F|I|B|C|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")
D_MODEL = 16


def tokens(smiles: str = FLUCONAZOLE) -> list[str]:
    return TOKEN.findall(smiles)


def attention_weights(smiles: str = FLUCONAZOLE, scaled: bool = True) -> np.ndarray:
    pieces = tokens(smiles)
    vocabulary = {t: i for i, t in enumerate(sorted(set(pieces)))}
    rng = np.random.default_rng(0)
    embedding = rng.normal(0, 1, (len(vocabulary), D_MODEL))
    positions = np.arange(len(pieces))[:, None] / (10000 ** (np.arange(0, D_MODEL, 2) / D_MODEL))
    encoding = np.zeros((len(pieces), D_MODEL))
    encoding[:, 0::2], encoding[:, 1::2] = np.sin(positions), np.cos(positions)
    x = embedding[[vocabulary[t] for t in pieces]] + encoding
    q, k = x @ rng.normal(0, 1 / np.sqrt(D_MODEL), (D_MODEL, D_MODEL)), x @ rng.normal(0, 1 / np.sqrt(D_MODEL), (D_MODEL, D_MODEL))
    scores = q @ k.T / (np.sqrt(D_MODEL) if scaled else 1.0)
    scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    return scores / scores.sum(axis=1, keepdims=True)


def attention_caption() -> str:
    scaled, unscaled = attention_weights(), attention_weights(scaled=False)
    return (f"Scaled dot-product self-attention over the {len(tokens())} tokens of the SMILES string of fluconazole. Every token is turned into a vector (an embedding plus a sinusoidal code of its position), projected into a query and a key, and the weight that token i gives to token j is the softmax over j of the dot product of "
            f"its query with their keys divided by the square root of the dimension ({D_MODEL}). The projections here are random with a fixed seed: the figure shows the computation, not what a trained model attends to (a trained model, ChemBERTa, is used in section 7.9). "
            f"Every row sums to one ({scaled.sum(axis=1).min():.3f} to {scaled.sum(axis=1).max():.3f}). Left: with the division; right: without it, the dot products grow with the dimension, the softmax saturates and each token puts almost all of its weight on one other token "
            f"(the largest weight in a row is {scaled.max(axis=1).mean():.2f} on average with the division and {unscaled.max(axis=1).mean():.2f} without); each panel is shaded up to its own largest weight.")


@figure("ml_attention", GROUP, "Self-attention over the tokens of a SMILES string", attention_caption, data=("the SMILES of fluconazole",), section="4.8")
def attention(theme_name):
    th = apply(theme_name)
    labels = tokens()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    cmap = LinearSegmentedColormap.from_list("a", [th["background"], th["blue"]])
    for ax, scaled, title in ((axes[0], True, "with the division by sqrt(d)"), (axes[1], False, "without it")):
        weights = attention_weights(scaled=scaled)
        image = ax.imshow(weights, cmap=cmap, vmin=0, vmax=weights.max())
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="attention weight")
        ax.set_xticks(range(len(labels)), labels, fontsize=6.5)
        ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
        ax.set(xlabel="token attended to (key)", ylabel="token asking (query)", title=title)
    fig.tight_layout()
    return fig


# -- principal components -------------------------------------------------------------------------------------------------------------------------

@cache
def component_study() -> dict:
    train = pd.read_csv(ROOT / TRAIN)
    store = np.load(ROOT / STORE, allow_pickle=True)
    smiles = list(store["smiles"])
    index = {s: i for i, s in enumerate(smiles)}
    labelled_rows = train[train["CYP3A4_is_TDI"].notna() & train["CYP3A4_pIC50_TDI_condition"].notna() & train["CYP3A4_pIC50_direct_inhibition"].notna()]
    rows = [index[s] for s in labelled_rows["SMILES"] if s in index]
    y = labelled_rows.loc[labelled_rows["SMILES"].isin(index), "CYP3A4_is_TDI"].astype(int).to_numpy()
    descriptors = StandardScaler().fit_transform(np.nan_to_num(store["rdkit_desc"][rows].astype(float)))
    keep = descriptors.std(axis=0) > 0
    pca = PCA(n_components=20, random_state=0).fit(descriptors[:, keep])
    embedding = store["chemeleon"][rows].astype(float)
    return {"explained": pca.explained_variance_ratio_, "scores": pca.transform(descriptors[:, keep])[:, :2], "embedding": embedding[:, :2], "y": y, "n": len(rows), "columns": int(keep.sum())}


def pca_caption() -> str:
    study = component_study()
    cumulative = np.cumsum(study["explained"])
    return (f"Principal component analysis of real CYP3A4 compounds. Left: the share of variance carried by each of the first 20 principal components of the {study['columns']} standardised RDKit descriptors of {study['n']} compounds (bars) and the running total (line): the first component carries "
            f"{study['explained'][0]:.0%} and ten components carry {cumulative[9]:.0%}, so a few directions summarise many correlated columns. Middle: the compounds on the first two components of the descriptors, time-dependent inhibitors in red; the two classes overlap almost completely, "
            "which is why no model of the descriptors alone is accurate. Right: the first two components of the frozen CheMeleon embedding of the same compounds (the block of the feature store keeps the embedding as 64 principal components); "
            "the same picture, a little more structured, which is the information the embedding adds. Principal components are fitted without any label.")


@figure("ml_pca", GROUP, "Principal components of the descriptors and of the CheMeleon embedding", pca_caption, data=(TRAIN, STORE), section="4.9")
def pca(theme_name):
    th = apply(theme_name)
    study = component_study()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    ax = axes[0]
    ax.bar(np.arange(1, 21), study["explained"], color=th["blue"])
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, 21), np.cumsum(study["explained"]), color=th["orange"], marker="o", ms=3)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("running total", color=th["orange"])
    ax.set(xlabel="principal component", ylabel="share of the variance", title="how many directions matter")
    tidy(ax, th)
    for ax, key, title in ((axes[1], "scores", "descriptors: PC1 and PC2"), (axes[2], "embedding", "CheMeleon embedding: PC1 and PC2")):
        points, y = study[key], study["y"]
        ax.scatter(points[y == 0, 0], points[y == 0, 1], s=5, color=th["blue"], alpha=0.35, linewidths=0, label="not TDI")
        ax.scatter(points[y == 1, 0], points[y == 1, 1], s=5, color=th["red"], alpha=0.5, linewidths=0, label="TDI")
        ax.set(xlabel="component 1", ylabel="component 2", title=title)
        tidy(ax, th)
    axes[1].legend(frameon=False, markerscale=3)
    fig.tight_layout()
    return fig


# -- the confusion matrix and the Matthews correlation coefficient ----------------------------------------------------------------------------------

@cache
def confusion_counts(isoform: str) -> dict:
    """The default stack at the called rate of the final build on each held-out outer fold of the 5-fold design (each fold calls the same top share): the mean of the weighted confusion matrices
    and the mean of the weighted MCC of the folds, which is how the final build reports it."""
    import json

    called = json.loads((ROOT / FINAL).read_text())["isoforms"][isoform]["called_rate"]
    cells, mccs, accuracies = [], [], []
    for y, score, w, _ in evaluation(isoform)[2]:
        predicted = score >= np.quantile(score, 1 - called)
        tp, fp = w[(y == 1) & predicted].sum(), w[(y == 0) & predicted].sum()
        fn, tn = w[(y == 1) & ~predicted].sum(), w[(y == 0) & ~predicted].sum()
        total = tp + fp + fn + tn
        cells.append((tp / total, fp / total, fn / total, tn / total))
        mccs.append((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        accuracies.append((tp + tn) / total)
    tp, fp, fn, tn = np.mean(cells, axis=0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "mcc": float(np.mean(mccs)), "accuracy": float(np.mean(accuracies)), "called": called, "folds": len(cells)}


def table_mcc(isoform: str) -> float:
    """The MCC of the default stack at z0 = 1.0 in the table of the final build (the rate is transferred inside every fold there)."""
    text = (ROOT / "data/tdi_track/outputs/final/final_report.md").read_text()
    block = re.search(rf"## {isoform}\n(.*?)(?=\n## |\Z)", text, re.DOTALL).group(1)
    row = next(line for line in block.splitlines() if line.startswith("| default |"))
    return float(row.strip("|").split("|")[3])


def confusion_caption() -> str:
    parts = []
    for iso in ("CYP3A4", "CYP2D6"):
        c = confusion_counts(iso)
        parts.append(f"{iso}: top {c['called']:.1%} called, accuracy {c['accuracy']:.2f}, MCC {c['mcc']:.3f} (the table of section 6 gives {table_mcc(iso):.3f}, where the rate is transferred inside each fold and not fixed), "
                     f"and calling everything negative would give accuracy {1 - c['tp'] - c['fn']:.2f} and MCC 0")
    return ("The confusion matrix of the default stack at the share of calls that the final build uses, from the held-out outer folds of the 5-fold design, as shares of the weighted scored population (the evaluation weights of the final build, confident labels at z0 = 1.0). "
            "Each cell is the mean over the outer folds of a share of all compounds, and the MCC in the title is the mean of the MCC of the folds (the way the final build reports it); the Matthews correlation coefficient of a fold is (TP x TN - FP x FN) divided by the square root of the product of the four marginal totals, so that it is zero for a model that predicts one class for everything and uses all four cells. "
            + "; ".join(parts) + ". Accuracy rewards the majority class on an imbalanced label; the MCC does not, which is why it is the metric of the challenge.")


@figure("ml_confusion_mcc", GROUP, "The confusion matrix and the MCC of the default stack", confusion_caption,
        data=("data/tdi_track/outputs/nested/<isoform>/plan.npz", "data/tdi_track/outputs/nested/<isoform>/cache", FINAL), section="4.10")
def confusion_mcc(theme_name):
    th = apply(theme_name)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    cmap = LinearSegmentedColormap.from_list("c", [th["background"], th["blue"]])
    for ax, iso in zip(axes, ("CYP3A4", "CYP2D6"), strict=True):
        c = confusion_counts(iso)
        grid = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=max(grid.max(), 0.5))
        names = np.array([["true negative", "false positive"], ["false negative", "true positive"]])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{names[i, j]}\n{grid[i, j]:.1%}", ha="center", va="center", color=th["text"] if grid[i, j] < 0.35 else th["background"], fontsize=10)
        ax.set_xticks([0, 1], ["called negative", "called positive"])
        ax.set_yticks([0, 1], ["not TDI", "TDI"])
        ax.set_title(f"{iso}: MCC {c['mcc']:.2f}, accuracy {c['accuracy']:.2f}")
    fig.tight_layout()
    return fig
