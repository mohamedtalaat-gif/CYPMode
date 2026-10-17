"""Concept figures of machine learning: a decision tree fitted to real compounds and gradient boosting stage by stage on a toy regression."""

from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import figure
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from style import apply, tidy

GROUP = "ml"
ROOT = Path(__file__).resolve().parents[3]
TRAIN = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
FEATURES = ("MolLogP", "TPSA")
DEPTH = 3
LEAF = 40


@cache
def descriptor_table() -> pd.DataFrame:
    train = pd.read_csv(ROOT / TRAIN)
    keep = train["CYP3A4_is_TDI"].notna() & train["CYP3A4_pIC50_TDI_condition"].notna() & train["CYP3A4_pIC50_direct_inhibition"].notna()
    table = train.loc[keep, ["SMILES", "CYP3A4_is_TDI"]].reset_index(drop=True)
    mols = [Chem.MolFromSmiles(s) for s in table["SMILES"]]
    for name in FEATURES:
        table[name] = [getattr(Descriptors, name)(m) for m in mols]
    table["label"] = table["CYP3A4_is_TDI"].astype(int)
    return table


def fitted_tree():
    table = descriptor_table()
    x, y = table[list(FEATURES)].to_numpy(), table["label"].to_numpy()
    tree = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=LEAF, random_state=0).fit(x, y)
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    held_out = cross_val_predict(DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=LEAF, random_state=0), x, y, cv=folds, method="predict_proba")[:, 1]
    return tree, x, y, roc_auc_score(y, held_out)


def tree_caption() -> str:
    tree, x, y, auc = fitted_tree()
    return (f"A decision tree of depth {DEPTH} (at least {LEAF} compounds in a leaf) fitted to the {len(y)} labelled CYP3A4 compounds using only two descriptors, the octanol-water partition coefficient "
            f"(MolLogP) and the polar surface area (TPSA). Left: the plane cut into the {tree.get_n_leaves()} rectangles the tree defines; a rectangle is shaded from blue (few time-dependent inhibitors in its leaf) to red (many) by the share of positives in its leaf, which is the probability the tree gives to every compound that falls in it; no leaf has a majority of positives, so a hard yes/no tree would answer no everywhere. "
            f"Right: the same tree as questions: each box asks whether one descriptor is below a threshold. Five-fold cross-validated AUC of this tree: {auc:.3f}. "
            f"A tree is easy to read and unstable (a slightly different sample gives different cuts), which is why the project uses forests and boosted trees of many of them.")


@figure("ml_decision_tree", GROUP, "A decision tree on real compounds", tree_caption, data=(TRAIN,), section="4.2")
def decision_tree(theme_name):
    th = apply(theme_name)
    tree, x, y, _ = fitted_tree()
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1, 2.2]})
    gx, gy = np.meshgrid(np.linspace(-2.5, 8, 300), np.linspace(0, 160, 300))
    zone = tree.predict_proba(np.c_[gx.ravel(), gy.ravel()])[:, 1].reshape(gx.shape)
    shade = LinearSegmentedColormap.from_list("positive_share", [th["blue"], th["red"]])
    a.contourf(gx, gy, zone, levels=np.linspace(0, 0.8, 17), cmap=shade, alpha=0.22, extend="both")
    a.scatter(x[y == 0, 0], x[y == 0, 1], s=6, color=th["blue"], alpha=0.5, linewidths=0, label="not time-dependent")
    a.scatter(x[y == 1, 0], x[y == 1, 1], s=6, color=th["red"], alpha=0.5, linewidths=0, label="time-dependent")
    a.set(xlabel="MolLogP", ylabel="TPSA (A^2)", xlim=(-2.5, 8), ylim=(0, 160))
    a.legend(frameon=False, fontsize=8, loc="upper right")
    tidy(a, th)
    plot_tree(tree, feature_names=list(FEATURES), class_names=["no", "yes"], filled=True, rounded=True, impurity=False, proportion=False, fontsize=8, ax=b)
    fig.tight_layout()
    return fig


NU = 0.3  # learning rate: the share of each new stump that is added
STAGES = (0, 1, 3, 10, 40)


@cache
def boosting_run():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 6, 120))
    y = np.sin(x) + 0.25 * rng.standard_normal(len(x))
    grid = np.linspace(0, 6, 300)
    prediction, grid_prediction = np.full(len(x), y.mean()), np.full(len(grid), y.mean())
    history = [(prediction.copy(), grid_prediction.copy(), y - prediction, np.zeros(len(grid)))]
    for _ in range(max(STAGES)):
        residual = y - prediction
        stump = DecisionTreeRegressor(max_depth=1).fit(x[:, None], residual)
        prediction = prediction + NU * stump.predict(x[:, None])
        grid_step = NU * stump.predict(grid[:, None])
        grid_prediction = grid_prediction + grid_step
        history.append((prediction.copy(), grid_prediction.copy(), y - prediction, grid_step))
    return x, y, grid, history


def boosting_caption() -> str:
    x, y, _, history = boosting_run()
    errors = [float(np.mean(h[2] ** 2)) for h in history]
    return (f"Gradient boosting on a toy regression (120 noisy points of sin x, drawn with a fixed seed; not project data). Stage 0 is the mean. Each stage fits a one-split tree (a stump) to the "
            f"residuals r_i = y_i - F(x_i), which are the negative gradient of the squared error, and adds a fraction nu = {NU} of it: F_m = F_(m-1) + nu h_m. Top row: the model after "
            f"{', '.join(str(s) for s in STAGES)} stumps; bottom row: the residuals it leaves. Mean squared error {errors[0]:.3f} at stage 0, {errors[10]:.3f} after 10 stumps, "
            f"{errors[40]:.3f} after 40; the noise floor of the data is about {0.25 ** 2:.3f}, which more stumps cannot go below without fitting the noise.")


@figure("ml_boosting_stages", GROUP, "Gradient boosting, stage by stage", boosting_caption, data=("synthetic regression with a fixed seed",), section="4.4", animation=lambda theme: boosting_animation(theme))
def boosting_stages(theme_name):
    th = apply(theme_name)
    x, y, grid, history = boosting_run()
    fig, axes = plt.subplots(2, len(STAGES), figsize=(13, 5.4), sharex=True)
    for col, stage in enumerate(STAGES):
        prediction, grid_prediction, residual, _ = history[stage]
        axes[0][col].plot(x, y, "o", color=th["muted"], ms=3, alpha=0.7)
        axes[0][col].plot(grid, grid_prediction, color=th["red"], lw=2.2)
        axes[0][col].set_title(f"after {stage} stump{'s' if stage != 1 else ''}", fontsize=10)
        axes[1][col].plot(x, residual, "o", color=th["blue"], ms=3, alpha=0.7)
        axes[1][col].axhline(0, color=th["text"], lw=1)
        axes[1][col].set_ylim(-1.6, 1.6)
        axes[0][col].set_ylim(-1.8, 1.8)
        for ax in (axes[0][col], axes[1][col]):
            tidy(ax, th)
    axes[0][0].set_ylabel("data and model")
    axes[1][0].set_ylabel("residuals")
    fig.tight_layout()
    return fig


def boosting_animation(theme_name: str) -> Path:
    th = apply(theme_name)
    x, y, grid, history = boosting_run()
    out = ROOT / "docs/figures" / GROUP / f"ml_boosting_stages-anim-{theme_name}.gif"
    frames = []
    for stage in range(max(STAGES) + 1):
        prediction, grid_prediction, residual, grid_step = history[stage]
        fig, (a, b) = plt.subplots(2, 1, figsize=(5.8, 5.2), dpi=110, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        a.plot(x, y, "o", color=th["muted"], ms=3, alpha=0.7)
        a.plot(grid, grid_prediction, color=th["red"], lw=2.2)
        a.set(ylim=(-1.8, 1.8), title=f"stage {stage}: mean squared error {np.mean(residual ** 2):.3f}")
        b.plot(x, residual, "o", color=th["blue"], ms=3, alpha=0.7)
        b.plot(grid, grid_step / NU if stage else np.zeros(len(grid)), color=th["green"], lw=1.6, label="stump fitted to these residuals" if stage else None)
        b.axhline(0, color=th["text"], lw=1)
        b.set(ylim=(-1.6, 1.6), ylabel="residual")
        for ax in (a, b):
            tidy(ax, th)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]))
        plt.close(fig)
    frames[0].save(out, save_all=True, append_images=frames[1:] + [frames[-1]] * 8, duration=120, loop=0)
    return out
