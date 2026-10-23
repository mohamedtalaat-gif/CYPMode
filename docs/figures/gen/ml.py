# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Figures about measuring honestly and about the final stack: nested-selection optimism, every stacked comparison, and the stack's
out-of-fold ROC, precision-recall and reliability curves rebuilt from the cached predictions."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import figure
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from style import apply, tidy

from gen._stack import evaluation

GROUP = "ml"
ROOT = Path(__file__).resolve().parents[3]
NESTED = "data/tdi_track/outputs/nested"
LOGS = "data/structure_track/outputs/cyp3a4/logs"
ISOFORMS = ("CYP3A4", "CYP2D6")
SELECTORS = {"best_single": "best single model", "one_se": "one-standard-error rule", "stack_logit": "stacked logistic blend"}


def optimism_caption() -> str:
    parts = []
    for iso in ISOFORMS:
        table = evaluation(iso)[1]
        for selector, label in SELECTORS.items():
            part = table[table["selector"] == selector]
            parts.append(f"{iso} {label}: inner {part['inner_score'].mean():.3f}, outer {part['outer_score'].mean():.3f}")
    return ("What each way of choosing a model reports about itself (inner score, measured on the data it was chosen on) against what the held-out outer folds then show "
            "(outer score): weighted AUC at z0 = 1.0, mean over the 15 outer folds of the 5-fold design, rebuilt from the cached predictions. " + "; ".join(parts) + ". "
            "A selector's inner score is not an estimate of its future score; only the outer score is reported. The size and even the sign of the gap depend on the weighting, "
            "which is why the comparison is made on outer folds that no choice has seen.")


@figure("ml_selection_optimism", GROUP, "What a model reports about itself and what held-out folds show", optimism_caption,
        data=(f"{NESTED}/<isoform>/plan.npz", f"{NESTED}/<isoform>/cache"), section="6.3")
def selection_optimism(theme_name):
    th = apply(theme_name)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, iso in zip(axes, ISOFORMS, strict=True):
        table = evaluation(iso)[1]
        selectors = list(SELECTORS)
        x = np.arange(len(selectors))
        inner = [table.loc[table["selector"] == s, "inner_score"] for s in selectors]
        outer = [table.loc[table["selector"] == s, "outer_score"] for s in selectors]
        ax.bar(x - 0.19, [v.mean() for v in inner], 0.36, yerr=[v.std(ddof=1) for v in inner], color=th["muted"], capsize=3, label="inner (chosen on)")
        ax.bar(x + 0.19, [v.mean() for v in outer], 0.36, yerr=[v.std(ddof=1) for v in outer], color=th["blue"], capsize=3, label="outer (held out)")
        ax.set_xticks(x, [SELECTORS[s].replace(" ", "\n", 1) for s in selectors], fontsize=8)
        ax.set(title=iso, ylim=(0.5, 0.95))
        tidy(ax, th, grid=False)
    axes[0].set_ylabel("weighted AUC (mean, bar = SD over outer folds)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


LOG_NAMES = {
    "physics": "physics_experiment_{iso}.log", "pair_desc": "pair_transfer_{iso}.log", "pair_physics": "pair_transfer_physics_{iso}.log", "mmp": "mmp_transfer_{iso}.log",
    "embeddings": "embeddings_{iso}.log", "arms_clip": "arms_experiment_{iso}.log", "committee": "committee_label_model_{iso}.log",
    "physics_compact": "physics_compact_{iso}.log", "physics_b": "physics_tierb_{iso}.log", "xiso": "xiso_transfer_{iso}.log",
    # the additions of the later rounds; a log that does not exist (yet) is skipped, so the plot grows as the queue finishes
    "screen": "aux_screen_{iso}.log", "emax": "aux_emax_{iso}.log", "pxr": "aux_pxr_{iso}.log", "gate_shift": "aux_gate_shift_{iso}.log", "no_direct": "aux_nodirect_{iso}.log",
    "basic_centre": "aux_basic_centre_{iso}.log", "ext_cont": "aux_ext_cont_{iso}.log", "ext_tox": "aux_ext_tox_{iso}.log", "ext_adme": "aux_ext_adme_{iso}.log", "atomjepa": "aux_atomjepa_{iso}.log", "pocket_cyp3a4": "aux_pocket_cyp3a4_{iso}.log", "pocket_cyp2d6": "aux_pocket_cyp2d6_{iso}.log",
    "tabicl": "aux_tabicl_{iso}.log", "tabpfn": "aux_tabpfn_{iso}.log", "tabicl_monroe": "aux_tabicl_monroe_{iso}.log", "tabicl_chemeleon": "aux_tabicl_chemeleon_{iso}.log", "tabpfn_chemeleon": "aux_tabpfn_chemeleon_{iso}.log", "multitask": "aux_mlp_{iso}.log", "multitask_aux": "aux_mlp_aux_{iso}.log",
    "chemprop": "aux_chemprop_{iso}.log", "chemeleon": "aux_chemeleon_{iso}.log", "neural_pair": "aux_neural_{iso}.log", "monroe": "monroe_embeddings_{iso}.log",
}
REPLICATE_LOG_NAMES = {"physics": "physics_experiment_f10_{iso}.log", "physics_compact": "physics_compact_f10_{iso}.log", "physics_b": "physics_tierb_f10_{iso}.log", "neural_pair": "aux_neural_f10_{iso}.log"}
STACKED = re.compile(r"z0=([\d.]+)\s+stacked base\+added vs base: AUC ([+-][\d.]+) \(p=([\d.e-]+)")


def stacked_comparisons(names=LOG_NAMES) -> pd.DataFrame:
    rows = []
    for idea, pattern in names.items():
        for iso in ISOFORMS:
            path = ROOT / LOGS / pattern.format(iso=iso)
            if not path.exists():
                continue
            for z0, diff, p in STACKED.findall(path.read_text()):
                rows.append({"idea": idea, "isoform": iso, "z0": float(z0), "auc_difference": float(diff), "p": float(p)})
    return pd.DataFrame(rows)


def forest_caption() -> str:
    five, ten = stacked_comparisons(), stacked_comparisons(REPLICATE_LOG_NAMES)
    both = lambda f: sum(1 for (_, _), g in f.groupby(["idea", "isoform"]) if len(g) == 2 and (g["p"] < 0.05).all() and (g["auc_difference"] > 0).all())
    return (f"Every stacked comparison logged by the experiments: the change in outer AUC when an idea's candidates are added to the default stack, for each isoform and each "
            f"confidence cut z0 (circle 1.0, square 1.96), with the p value printed in the experiment's log beside each marker. Top row: the 5-fold design, {len(five)} comparisons from "
            f"{five['idea'].nunique()} ideas; bottom row: the 10-fold replicate of the physics ideas, {len(ten)} comparisons. An idea passes its pre-declared gate only when both z0 "
            f"cuts have p < 0.05 and a positive difference for the same isoform (filled green markers): {both(five)} of {five.groupby(['idea', 'isoform']).ngroups} in the 5-fold design "
            f"and {both(ten)} of {ten.groupby(['idea', 'isoform']).ngroups} in the replicate.")


@figure("ml_gate_forest", GROUP, "Every stacked comparison of the project in one plot", forest_caption,
        data=tuple(f"{LOGS}/{p.format(iso='<isoform>')}" for p in {**LOG_NAMES, **REPLICATE_LOG_NAMES}.values()), section="6.6")
def gate_forest(theme_name):
    th = apply(theme_name)
    five, ten = stacked_comparisons(), stacked_comparisons(REPLICATE_LOG_NAMES)
    ideas = list(dict.fromkeys(five["idea"]))
    ideas_ten = list(dict.fromkeys(ten["idea"]))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 0.5 * (len(ideas) + len(ideas_ten)) + 2.6), sharex=True,
                             gridspec_kw={"height_ratios": [len(ideas) + 1, len(ideas_ten) + 1]})
    for r, (frame, names, design) in enumerate(((five, ideas, "5-fold design"), (ten, ideas_ten, "10-fold replicate"))):
        for ax, iso in zip(axes[r], ISOFORMS, strict=True):
            for i, idea in enumerate(names):
                for z0, dy, marker in ((1.0, -0.17, "o"), (1.96, 0.17, "s")):
                    row = frame[(frame["idea"] == idea) & (frame["isoform"] == iso) & (frame["z0"] == z0)]
                    if row.empty:
                        continue
                    value, p = float(row["auc_difference"].iloc[0]), float(row["p"].iloc[0])
                    passed = p < 0.05 and value > 0
                    color = th["green"] if passed else th["muted"]
                    ax.plot(value, i + dy, marker, color=color, mfc=color if passed else "none", ms=6)
                    ax.text(value + 0.00022, i + dy, f"{p:.2f}", va="center", fontsize=6.5, color=th["muted"])
            ax.axvline(0, color=th["text"], lw=1)
            ax.set_yticks(range(len(names)), names)
            ax.set_ylim(len(names) - 0.4, -0.6)
            ax.set_title(f"{iso}: {design}", fontsize=10)
            ax.set_xlim(-0.0085, 0.0135)
            ax.set_xticks([-0.008, -0.004, 0, 0.004, 0.008, 0.012])
            tidy(ax, th)
    for ax in axes[1]:
        ax.set_xlabel("change in outer AUC when added to the default stack")
    axes[0][0].plot([], [], "o", color=th["muted"], mfc="none", label="z0 = 1.0")
    axes[0][0].plot([], [], "s", color=th["muted"], mfc="none", label="z0 = 1.96")
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def curves_caption() -> str:
    rows = []
    for iso in ISOFORMS:
        folds = evaluation(iso)[2]
        rows.append(f"{iso} mean outer AUC {np.mean([weighted_auc(y, s, w) for y, s, w, _ in folds]):.3f}")
    return ("Receiver-operating-characteristic, precision-recall and reliability curves of the default stack on the held-out outer folds of the 5-fold design, rebuilt from the cached "
            "predictions with the evaluation weights of the final build (confident labels at z0 = 1.0, test-like similarity, prevalence shifted to the test set); one thin line per outer fold, "
            "the mean as the thick line, " + "; ".join(rows) + ", equal to the final build's numbers. The dashed line of the precision-recall panel is the weighted prevalence. "
            "The right panels show the weighted positive rate in each tenth of the stack's scores, pooled over the outer folds. The raw stack score is not a probability "
            "(the audit measures the CYP2D6 raw probabilities as unusable, which is why the submitted probabilities are isotonic-calibrated), so ranking is shown rather than calibration.")


def weighted_auc(y, score, w):
    return roc_auc_score(y, score, sample_weight=w)


@figure("ml_stack_curves", GROUP, "ROC, precision-recall and score-decile lift of the final stack", curves_caption,
        data=(f"{NESTED}/<isoform>/plan.npz", f"{NESTED}/<isoform>/cache"), section="4.9")
def stack_curves(theme_name):
    th = apply(theme_name)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    grid = np.linspace(0, 1, 201)
    for row, iso in enumerate(ISOFORMS):
        folds = evaluation(iso)[2]
        roc_ax, pr_ax, cal_ax = axes[row]
        tprs, precisions, aucs, aps = [], [], [], []
        pooled = []
        for y, score, w, w_plain in folds:
            fpr, tpr, _ = roc_curve(y, score, sample_weight=w)
            roc_ax.plot(fpr, tpr, color=th["blue"], alpha=0.18, lw=1)
            tprs.append(np.interp(grid, fpr, tpr))
            precision, recall, _ = precision_recall_curve(y, score, sample_weight=w)
            order = np.argsort(recall)
            pr_ax.plot(recall, precision, color=th["red"], alpha=0.18, lw=1)
            precisions.append(np.interp(grid, recall[order], precision[order]))
            aucs.append(roc_auc_score(y, score, sample_weight=w))
            aps.append(average_precision_score(y, score, sample_weight=w))
            pooled.append((pd.Series(score).rank(pct=True).to_numpy(), y, w))
        roc_ax.plot(grid, np.mean(tprs, 0), color=th["blue"], lw=2.6, label=f"AUC {np.mean(aucs):.3f}")
        roc_ax.plot([0, 1], [0, 1], color=th["muted"], ls="--", lw=1)
        roc_ax.set(title=f"{iso}: ROC", xlabel="false-positive rate", ylabel="true-positive rate")
        roc_ax.legend(frameon=False, loc="lower right")
        pr_ax.plot(grid, np.mean(precisions, 0), color=th["red"], lw=2.6, label=f"AP {np.mean(aps):.3f}")
        weighted_prevalence = np.mean([np.average(y, weights=w) for y, _, w, _ in folds])
        pr_ax.axhline(weighted_prevalence, color=th["muted"], ls="--", lw=1)
        pr_ax.set(title=f"{iso}: precision-recall", xlabel="recall", ylabel="precision", ylim=(0, 1.02))
        pr_ax.legend(frameon=False, loc="upper right")
        quantile, truth, weight = (np.concatenate(v) for v in zip(*pooled, strict=True))
        decile = np.minimum((quantile * 10).astype(int), 9)
        rate = [np.average(truth[decile == b], weights=weight[decile == b]) for b in range(10)]
        cal_ax.bar(np.arange(10) + 0.5, rate, 0.8, color=th["green"])
        cal_ax.axhline(np.average(truth, weights=weight), color=th["muted"], ls="--", lw=1)
        cal_ax.set(title=f"{iso}: positive rate by score decile", xlabel="score decile (10 = highest scores)", ylabel="weighted positive rate", xticks=np.arange(10) + 0.5, xticklabels=range(1, 11))
        for ax in (roc_ax, pr_ax, cal_ax):
            tidy(ax, th)
    fig.tight_layout()
    return fig
