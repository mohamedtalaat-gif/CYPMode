# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The overview figures that the plan listed and that the project lacked: the ledger of every idea as one heat map, the final results on one page, the history of the repository, Butina clusters, the evaluation weights,
the rule of the gate as a flow chart, the staircase M0 to M3, the calibration methods, the sensitivity of the selection-aware model, the correlation of the assay arms, the label as a Gaussian, the three structure engines, the composition of
the pool of the final recipe (held back), the map of the repository, the flow of the data and the packages of the code. Each is drawn from the repository's own files and logs."""

import re
import subprocess
from collections import Counter
from functools import cache
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import figure
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from scipy.stats import norm, spearmanr
from style import apply, tidy

from gen._draw import arrow, box, canvas

ROOT = Path(__file__).resolve().parents[3]
TRAIN = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
LOGS = "data/structure_track/outputs/cyp3a4/logs"
OUT = "data/structure_track/outputs/cyp3a4"
COMPARISONS = "data/tdi_track/outputs/comparisons"
FINAL = "data/tdi_track/outputs/final"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"
ISOFORMS = ("CYP3A4", "CYP2D6")
LOG2 = float(np.log10(2))


# -- the ledger as a heat map -----------------------------------------------------------------------------------------------------------------------

@cache
def ledger_matrix() -> tuple[list[str], list[str], np.ndarray, list[str]]:
    """Rows: the families of ideas grouped by kind; columns: CYP3A4, CYP2D6, structure; cell 2 = a MET verdict, 1 = only NOT MET verdicts, 0 = not tested there."""
    from cypmode.audit.docs_state import TITLES
    from cypmode.audit.ledger_kinds import KINDS, ledger_rows

    ledger = ledger_rows()
    rows, kinds, cells = [], [], []
    for kind, stems in KINDS.items():
        for stem in stems:
            part = ledger[ledger["stem"] == stem]
            if part.empty:
                continue
            row = []
            for column in ("CYP3A4", "CYP2D6", None):
                if column is None:
                    chosen = part[~part["log"].str.contains("CYP3A4|CYP2D6")]
                else:
                    chosen = part[part["log"].str.contains(column)]
                row.append(0 if chosen.empty else 2 if (chosen["verdict"] == "MET").any() else 1)
            rows.append(TITLES.get(stem, stem.replace("_", " ")))
            kinds.append(kind)
            cells.append(row)
    return rows, ["CYP3A4", "CYP2D6", "structure"], np.array(cells), kinds


def ledger_caption() -> str:
    rows, _columns, cells, _kinds = ledger_matrix()
    return (f"Every idea that was put to a pre-registered gate, one row per family and one column per place it was tried (CYP3A4, CYP2D6, the structure track): green where a verdict was MET, grey where every verdict was NOT MET, empty where the idea was not tried. "
            f"{len(rows)} families, {int((cells == 2).sum())} cells with a MET verdict and {int((cells == 1).sum())} with none, grouped by the kind of idea. A family that is MET in one cell and NOT MET in the other is a result for one enzyme only. "
            "The count of verdicts behind the cells is the one count that the method report, the README and the film quote.")


@figure("arch_ledger_heatmap", "architecture", "Every idea by place and verdict", ledger_caption, data=(f"{LOGS}/*.log",), section="9.11")
def ledger_heatmap(theme_name):
    th = apply(theme_name)
    rows, columns, cells, kinds = ledger_matrix()
    fig, ax = plt.subplots(figsize=(7.4, 0.24 * len(rows) + 1.8))
    ax.imshow(cells, cmap=ListedColormap([th["background"], th["muted"], th["green"]]), vmin=0, vmax=2, aspect="auto")
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    ax.set_xticks(range(len(columns)), columns)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-0.5, len(columns)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=th["grid"], linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    boundaries = [i for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1]]
    for b in boundaries:
        ax.axhline(b - 0.5, color=th["text"], lw=1.2)
    first = 0
    for end in [*boundaries, len(kinds)]:
        ax.text(len(columns) - 0.45, (first + end - 1) / 2, kinds[first], rotation=270, va="center", ha="left", fontsize=7.5, color=th["muted"])
        first = end
    ax.legend(handles=[Patch(color=th["green"], label="MET"), Patch(color=th["muted"], label="NOT MET"), Patch(facecolor=th["background"], edgecolor=th["grid"], label="not tried")], frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.06 - 1.0 / len(rows)), ncol=3)
    fig.tight_layout()
    return fig


# -- the final results on one page --------------------------------------------------------------------------------------------------------------------

def final_tables() -> dict:
    out = {}
    for iso in ISOFORMS:
        choice = pd.read_csv(ROOT / FINAL / f"{iso}_stack_choice.csv").set_index("stack")
        import json

        manifest = json.loads((ROOT / FINAL / "manifest.json").read_text())["isoforms"][iso]
        out[iso] = {"default": choice.loc["default"], "final": choice.loc[manifest["stack"]], "name": manifest["stack"], "n": manifest["n_candidates"], "called": manifest["called_rate"], "prevalence": manifest["target_prevalence"]}
    return out


def final_caption() -> str:
    tables = final_tables()
    import json

    structure = json.loads((ROOT / OUT / "final/manifest.json").read_text())
    margins = [lig["distance_margin"] for lig in structure["ligands"]]
    parts = [f"{iso}: AUC {t['final']['auc_z1.0']:.3f} and MCC {t['final']['mcc_z1.0']:.3f} at z0 = 1.0 ({t['final']['auc_z1.96']:.3f} and {t['final']['mcc_z1.96']:.3f} at z0 = 1.96), against {t['default']['auc_z1.0']:.3f} and {t['default']['mcc_z1.0']:.3f} for the default stack" for iso, t in tables.items()]
    return ("The results of the project on one page, all of them estimates on the training data and none a score on the hidden labels. Left and middle: the nested cross-validated AUC and MCC of the final stack of each enzyme next to the default stack, at the two confidence cuts of the labels "
            "(the MCC is at the rate transferred from the scored prevalence). " + "; ".join(parts) + ". Right: the structure track in numbers, "
            f"{len(structure['ligands'])} ligands, {sum(lig['valid'] for lig in structure['ligands'])} poses passing all 22 PoseBusters checks and {sum(m >= 0.8 for m in margins)} with a margin of 0.8 or more, "
            f"the relaxation adopted for {structure['relaxation']['adopted']} of {structure['relaxation']['of']} poses.")


@figure("arch_final_results", "architecture", "The final results on one page", final_caption, data=(f"{FINAL}/*_stack_choice.csv", f"{FINAL}/manifest.json", f"{OUT}/final/manifest.json"), section="8.7")
def final_results(theme_name):
    import json

    th = apply(theme_name)
    tables = final_tables()
    structure = json.loads((ROOT / OUT / "final/manifest.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, metric, title in ((axes[0], "auc", "AUC of the nested design"), (axes[1], "mcc", "MCC at the transferred rate")):
        positions, labels = [], []
        for i, iso in enumerate(ISOFORMS):
            for j, z0 in enumerate(("1.0", "1.96")):
                x = i * 3 + j * 1.1
                for k, (which, color) in enumerate((("default", th["muted"]), ("final", th["blue"]))):
                    value = float(tables[iso][which][f"{metric}_z{z0}"])
                    ax.bar(x + (k - 0.5) * 0.4, value, 0.38, color=color, label=which if (i, j) == (0, 0) else None)
                    ax.text(x + (k - 0.5) * 0.4, value + 0.005, f"{value:.2f}", ha="center", fontsize=7, color=th["text"])
                positions.append(x)
                labels.append(f"{iso}\nz0 = {z0}")
        ax.set_xticks(positions, labels, fontsize=7.5)
        ax.set(title=title, ylim=(0, 1 if metric == "auc" else 0.65))
        tidy(ax, th, grid=False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    ax = axes[2]
    margins = [lig["distance_margin"] for lig in structure["ligands"]]
    facts = [("ligands", len(margins)), ("pass all 22 checks", sum(lig["valid"] for lig in structure["ligands"])), ("margin of 0.8 or more", sum(m >= 0.8 for m in margins)), ("relaxation adopted", structure["relaxation"]["adopted"])]
    ax.barh(range(len(facts)), [v for _, v in facts], color=th["green"])
    for i, (name, value) in enumerate(facts):
        ax.text(0.3, i, f"{name}: {value}", va="center", fontsize=9, color=th["background"] if value > 10 else th["text"])
    ax.set_yticks([])
    ax.set(title="structure track (20 ligands)", xlim=(0, len(margins) + 1))
    ax.invert_yaxis()
    tidy(ax, th, grid=False)
    fig.tight_layout()
    return fig


# -- the history of the repository --------------------------------------------------------------------------------------------------------------------

@cache
def commit_days() -> pd.Series:
    log = subprocess.run(["git", "log", "--format=%ad", "--date=short"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    return pd.Series(Counter(log)).sort_index()


@cache
def history_head() -> str:
    """The commit that the history was counted at, so that a count printed in a figure names the state it describes."""
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def timeline_caption() -> str:
    days = commit_days()
    return (f"The history of the repository as the number of commits of each day, from the first commit on {days.index[0]} to the last on {days.index[-1]}: {int(days.sum())} commits on {len(days)} days up to commit {history_head()}, the busiest day with {int(days.max())} ({days.idxmax()}). "
            "The history is kept as it is, never squashed, so that the date of every declared gate stays checkable against the date of the result that it judged; the commit messages say what each commit declared or found.")


@figure("eng_commit_timeline", "engineering", "The history of the repository", timeline_caption, data=("git log",), section="12.4")
def commit_timeline(theme_name):
    th = apply(theme_name)
    days = commit_days()
    index = pd.date_range(days.index[0], days.index[-1])
    counts = days.reindex(index.strftime("%Y-%m-%d"), fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(range(len(counts)), counts.to_numpy(), color=th["blue"], width=0.85)
    ticks = list(range(0, len(counts), max(1, len(counts) // 10)))
    ax.set_xticks(ticks, [counts.index[i][5:] for i in ticks], fontsize=8)
    ax.set(xlabel="day (month-day)", ylabel="commits", title=f"{int(counts.sum())} commits up to {history_head()}")
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- Butina clusters ------------------------------------------------------------------------------------------------------------------------------

@cache
def butina_study() -> dict:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.ML.Cluster import Butina

    smiles = pd.read_csv(ROOT / TRAIN)["SMILES"].sample(1500, random_state=0).tolist()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
    distances = []
    for i in range(1, len(fps)):
        distances.extend(1 - np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])))
    clusters = Butina.ClusterData(distances, len(fps), 0.65, isDistData=True)
    sizes = np.array([len(c) for c in clusters])
    biggest = clusters[int(np.argmax(sizes))]
    return {"sizes": sizes, "mols": [mols[i] for i in biggest[:8]], "n": len(fps), "cutoff": 0.65}


def butina_caption() -> str:
    study = butina_study()
    sizes = study["sizes"]
    return (f"Butina clustering of {study['n']} random training compounds: compounds closer than a Tanimoto distance of {study['cutoff']} (similarity above {1 - study['cutoff']:.2f}) to the centre of a cluster join it, the centres being chosen greedily by the number of neighbours. "
            f"Left: the sizes of the {len(sizes)} clusters; {int((sizes == 1).sum())} are single compounds and the largest holds {int(sizes.max())}. Right: eight members of the largest cluster, the chemical series that a random split of the data would spread over training and test, "
            "which is why the project also evaluates by similarity to the test structures and not by random folds alone.")


@figure("mol_butina_clusters", "molecules", "Butina clusters of the training compounds", butina_caption, data=(TRAIN,), section="3.6")
def butina_clusters(theme_name):
    from rdkit.Chem import Draw

    th = apply(theme_name)
    study = butina_study()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    ax.hist(study["sizes"], bins=np.unique(np.round(np.geomspace(1, study["sizes"].max() + 1, 25))), color=th["blue"])
    ax.set(xscale="log", yscale="log", xlabel="compounds in a cluster", ylabel="clusters", title="most clusters are small, a few are series")
    tidy(ax, th)
    ax = axes[1]
    image = Draw.MolsToGridImage(study["mols"], molsPerRow=4, subImgSize=(210, 170), useSVG=False)
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("eight members of the largest cluster")
    fig.tight_layout()
    return fig


# -- the evaluation weights --------------------------------------------------------------------------------------------------------------------------

def weight_caption() -> str:
    from gen._stack import evaluation

    parts = []
    for iso in ISOFORMS:
        folds = evaluation(iso)[2]
        w = np.concatenate([f[2] / f[2].mean() for f in folds])
        plain = np.concatenate([f[3] / f[3].mean() for f in folds])
        parts.append(f"{iso}: the top tenth of the compounds carries {np.sort(w)[::-1][: len(w) // 10].sum() / w.sum():.0%} of the weight (with the prevalence shift) and {np.sort(plain)[::-1][: len(plain) // 10].sum() / plain.sum():.0%} without it")
    return ("The weights that make a held-out fold resemble the scored population: a compound counts in proportion to its similarity to the blinded test structures, to the prevalence of its class in the test set, and only if its label is confident. "
            "For every outer fold of the 5-fold design the weights are scaled to a mean of one, and the figure shows their distribution for the positives and the negatives, with and without the shift of the prevalence. " + "; ".join(parts) + ". "
            "A weighted score is therefore an estimate for the scored population and not for the training set, and its effective sample size is smaller than the number of rows.")


@figure("ml_evaluation_weights", "ml", "The weights that make a fold resemble the test", weight_caption, data=("data/tdi_track/outputs/nested/<isoform>/plan.npz", "data/tdi_track/outputs/nested/<isoform>/cache"), section="6.4")
def evaluation_weights(theme_name):
    from gen._stack import evaluation

    th = apply(theme_name)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, iso in zip(axes, ISOFORMS, strict=True):
        folds = evaluation(iso)[2]
        y = np.concatenate([f[0] for f in folds])
        for column, label, color in ((2, "with the prevalence shift", th["blue"]), (3, "similarity and confidence only", th["muted"])):
            w = np.concatenate([f[column] / f[column].mean() for f in folds])
            for cls, style in ((1, "-"), (0, "--")):
                ax.hist(w[y == cls], bins=np.geomspace(max(w.min(), 1e-3), w.max(), 40), histtype="step", color=color, ls=style, lw=1.6, label=f"{label}, {'positives' if cls else 'negatives'}")
        ax.set(xscale="log", xlabel="weight of a compound (mean 1 in each fold)", title=iso)
        tidy(ax, th)
    axes[0].set_ylabel("compounds")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


# -- the rule of the gate as a flow chart -----------------------------------------------------------------------------------------------------------

def gate_caption() -> str:
    from cypmode.challenge.final_tdi_build import CONFIRMED_GROUPS, Z_CUTS

    return (f"The rule by which an added group joins a stack, as a flow chart; the final build reads the verdict lines of the logs and applies exactly this and nothing else. A group is tested alone and inside the stack at both confidence cuts (z0 = {' and '.join(str(z) for z in Z_CUTS)}), "
            "each time with the corrected resampled t against the stack without it; it must have p < 0.05 and a positive difference at both cuts, both alone and inside the stack. "
            f"A group that was found after looking at the 15-fold results ({', '.join(CONFIRMED_GROUPS)}) must in addition be replicated on ten outer folds, a design declared before any ten-fold number existed. "
            "A group that fails is written down with its numbers and is not tuned until it passes; the stack is chosen among the groups that pass.")


@figure("ml_gate_flow", "ml", "The gate: how a group joins a stack", gate_caption, data=("cypmode/challenge/final_tdi_build.py",), section="6.7")
def gate_flow(theme_name):
    th = apply(theme_name)
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    canvas(ax, (0, 12), (0, 5))
    steps = [(1.3, 3, "an idea", "written down with its rule before it runs", th["panel"]), (3.9, 3, "alone", "p < 0.05 and a gain at z0 = 1.0 and 1.96", th["panel"]), (6.5, 3, "inside the stack", "the same, against the stack without it", th["panel"]),
             (9.2, 3, "found after looking?", "then a replicate on ten outer folds must be MET too", th["panel"])]
    for x, y, title, text, fill in steps:
        box(ax, x, y, 2.3, 1.7, text, fill, th["muted"], th["text"], size=8, wrap=20, title=title)
    for a, b in pairwise(steps):
        arrow(ax, (a[0] + 1.18, 3), (b[0] - 1.18, 3), th["text"])
    box(ax, 9.2, 0.8, 3.0, 1.0, "joins the stack; the final stack is chosen among the groups that joined", th["green"], th["green"], th["background"], size=8, wrap=34, title="PASS")
    box(ax, 5.2, 0.8, 3.2, 1.0, "a measured negative result in the ledger, not tuned until it passes", th["red"], th["red"], th["background"], size=8, wrap=34, title="FAIL")
    arrow(ax, (9.2, 2.1), (9.2, 1.32), th["green"])
    arrow(ax, (3.9, 2.1), (4.5, 1.32), th["red"], ls="--")
    arrow(ax, (6.5, 2.1), (5.9, 1.32), th["red"], ls="--")
    arrow(ax, (8.4, 2.1), (6.8, 1.15), th["red"], ls="--", rad=0.15)
    ax.text(0.2, 4.7, "any failed step ends the test: the idea is a negative result", color=th["red"], fontsize=9)
    fig.tight_layout()
    return fig


# -- the staircase M0 to M3 --------------------------------------------------------------------------------------------------------------------------

def ablation_caption() -> str:
    table = pd.read_csv(ROOT / COMPARISONS / "ablation_m0_m3.csv")
    mean = table.groupby(["isoform", "model"])["mcc"].mean()
    stages = ("M0", "M1", "M2", "M3")
    by_iso = {iso: ", ".join(f"{m} {mean[(iso, m)]:.3f}" for m in stages) for iso in ISOFORMS}
    return (f"The first staircase of features, from the run of 2026-09-25 (5-fold cross-validation, threshold 0.5, the population as first built): M0 the RDKit descriptors, M1 with the Type II motif flag, M2 with the trialkylamine flag, M3 with a Morgan fingerprint. "
            f"MCC by stage, mean over the five folds: CYP3A4 {by_iso['CYP3A4']}; CYP2D6 {by_iso['CYP2D6']}. "
            "The steps are inside the spread of the folds, so the staircase shows what was tried and not a ranking; the M2 and M3 stages of this run kept the amine flag, which the specification later corrected (Tier 2), and the final model rests on none of these stages.")


@figure("ml_m0_m3_ablation", "ml", "The staircase M0 to M3", ablation_caption, data=(f"{COMPARISONS}/ablation_m0_m3.csv",), section="7.2")
def ablation(theme_name):
    th = apply(theme_name)
    table = pd.read_csv(ROOT / COMPARISONS / "ablation_m0_m3.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    for k, (iso, color) in enumerate((("CYP3A4", th["blue"]), ("CYP2D6", th["red"]))):
        part = table[table["isoform"] == iso].groupby("model")["mcc"].agg(["mean", "std"])
        x = np.arange(len(part)) + (k - 0.5) * 0.36
        ax.bar(x, part["mean"], 0.34, yerr=part["std"], color=color, capsize=3, label=iso)
    ax.set_xticks(range(4), ["M0 descriptors", "M1 + motif", "M2 + amine flag", "M3 + Morgan"])
    ax.set(ylabel="MCC (5-fold, mean and spread)", title="the first staircase of features")
    ax.legend(frameon=False)
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- the calibration methods -------------------------------------------------------------------------------------------------------------------------

CALIBRATION = re.compile(r"^\s*(CYP\w+)\s+(raw|platt|isotonic)\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(True|False)", re.MULTILINE)


@cache
def calibration_rows() -> pd.DataFrame:
    text = (ROOT / LOGS / "calibration.log").read_text(errors="ignore")
    return pd.DataFrame([{"isoform": i, "method": m, "ece": float(e), "slope": float(s), "met": g == "True"} for i, m, e, _b, s, _c, _p, _o, g in CALIBRATION.findall(text)])


def calibration_caption() -> str:
    table = calibration_rows()
    parts = [f"{iso}: " + ", ".join(f"{r.method} ECE {r.ece:.3f} slope {r.slope:.2f}" for r in table[table["isoform"] == iso].itertuples()) for iso in ISOFORMS]
    return ("How well the output of the stack can be read as a probability, for the three ways of turning a score into one (the raw output, Platt scaling and isotonic regression), measured on held-out folds. Left: the expected calibration error (the gap between the predicted and the observed rate of positives, "
            "binned); right: the calibration slope, which is 1 when a predicted probability of 0.3 means that 30% are positive. The gate asks for an error of at most 0.05 and a slope in [0.8, 1.2]; a method that meets it is filled. " + "; ".join(parts) + ". "
            "The raw CYP2D6 output is not a probability at all, and isotonic regression is the method that meets the gate for both enzymes, which is why the published probabilities pass through it.")


@figure("ml_calibration_methods", "ml", "Raw, Platt and isotonic calibration", calibration_caption, data=(f"{LOGS}/calibration.log",), section="8.5")
def calibration_methods(theme_name):
    th = apply(theme_name)
    table = calibration_rows()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    methods = ["raw", "platt", "isotonic"]
    for ax, column, title in ((axes[0], "ece", "expected calibration error"), (axes[1], "slope", "calibration slope")):
        for k, (iso, color) in enumerate((("CYP3A4", th["blue"]), ("CYP2D6", th["red"]))):
            part = table[table["isoform"] == iso].set_index("method").loc[methods]
            x = np.arange(3) + (k - 0.5) * 0.36
            ax.bar(x, part[column], 0.34, color=color, label=iso, alpha=1.0, edgecolor=[th["text"] if met else "none" for met in part["met"]], linewidth=1.6)
        if column == "ece":
            ax.axhline(0.05, color=th["gold"], ls="--", lw=1.2)
            ax.set_yscale("symlog", linthresh=0.05)
        else:
            ax.axhspan(0.8, 1.2, color=th["green"], alpha=0.15)
        ax.set_xticks(range(3), methods)
        ax.set(title=title)
        tidy(ax, th)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# -- the sensitivity of the selection-aware model ---------------------------------------------------------------------------------------------

SENSITIVITY = re.compile(r"z0=([\d.]+)\s+non-hit weight ([\d.]+)\s*\|\s*S0 AUC ([\d.]+) MCC ([+-][\d.]+)\s*\|\s*S1 AUC ([\d.]+) MCC ([+-][\d.]+)\s*\|\s*S2 AUC ([\d.]+) MCC ([+-][\d.]+)\s*\|\s*S3 AUC ([\d.]+) MCC ([+-][\d.]+)")
DEPLOY = re.compile(r"w=([\d.]+)\s+hits_\s+([\d.]+)\|([+-][\d.]+)\s+selec\s+([\d.]+)\|([+-][\d.]+)\s+blend\s+([\d.]+)\|([+-][\d.]+)")


@cache
def selection_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    sensitivity = (ROOT / LOGS / "population_sensitivity_panel_CYP2D6_r3.log").read_text(errors="ignore")
    rows = [{"z0": float(z), "w": float(w), **{f"S{i}": float(a) for i, a in enumerate((s0, s1, s2, s3))}} for z, w, s0, _m0, s1, _m1, s2, _m2, s3, _m3 in SENSITIVITY.findall(sensitivity)]
    decision = (ROOT / LOGS / "selection_decision_CYP2D6.log").read_text(errors="ignore")
    first = decision.split("z0 = 1.96")[0]
    deploy = [{"w": float(w), "hits-only": float(a), "selection-aware": float(b), "blend": float(c)} for w, a, _am, b, _bm, c, _cm in DEPLOY.findall(first)]
    return pd.DataFrame(rows), pd.DataFrame(deploy)


def selection_caption() -> str:
    sensitivity, deploy = selection_tables()
    low, high = deploy.iloc[0], deploy.iloc[-1]
    return (f"How much the selection-aware CYP2D6 model gains depends on how many never-promoted compounds the scored set holds, a share that nobody knows for the test set. The horizontal axis is the weight w of such compounds in the scored population (0: none, 1: as many as in the training panel). "
            f"Left: the outer AUC of the hits-only stack (S0) and of the three versions of the selection-aware model (S1 to S3, fitted as the training panel is) against w at the confidence cut z0 = 1.0, from the sensitivity runs: the hits-only stack leads when w is small and is behind S3 by {sensitivity[sensitivity['z0'] == 1.0].iloc[-1]['S3'] - sensitivity[sensitivity['z0'] == 1.0].iloc[-1]['S0']:.3f} AUC when w = 1. "
            f"Right: the deployment table of the registered decision (stacks fitted under w = 1.0, the top share of the final build called): the hits-only file against the selection-aware file and the rank blend; at w = {low['w']} the AUC is {low['hits-only']:.3f}, {low['selection-aware']:.3f} and {low['blend']:.3f}, at w = {high['w']} it is {high['hits-only']:.3f}, {high['selection-aware']:.3f} and {high['blend']:.3f}. "
            "The two files each win in one regime; the registered rule, which looks at the loss in the regime where the selection-aware file loses, gives the hits-only file.")


@figure("ml_selection_sensitivity", "ml", "The selection-aware model against the share of never-promoted compounds", selection_caption, data=(f"{LOGS}/population_sensitivity_panel_CYP2D6_r3.log", f"{LOGS}/selection_decision_CYP2D6.log"), section="9.1")
def selection_sensitivity(theme_name):
    th = apply(theme_name)
    sensitivity, deploy = selection_tables()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    ax = axes[0]
    part = sensitivity[sensitivity["z0"] == 1.0]
    for name, color in (("S0", th["muted"]), ("S1", th["blue"]), ("S2", th["teal"]), ("S3", th["red"])):
        ax.plot(part["w"], part[name], "o-", color=color, label={"S0": "S0 hits-only", "S1": "S1 + gate", "S2": "S2 + two-part", "S3": "S3 + end-to-end"}[name])
    ax.set(xlabel="weight of the never-promoted compounds in the scored set (w)", ylabel="outer AUC", title="sensitivity runs: fitted as the panel is")
    ax.legend(frameon=False, fontsize=8)
    tidy(ax, th)
    ax = axes[1]
    for name, color in (("hits-only", th["muted"]), ("selection-aware", th["red"]), ("blend", th["teal"])):
        ax.plot(deploy["w"], deploy[name], "o-", color=color, label=name)
    ax.set(xlabel="weight of the never-promoted compounds in the scored set (w)", title="deployment table of the registered decision")
    ax.legend(frameon=False, fontsize=8)
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- the correlation of the assay arms --------------------------------------------------------------------------------------------------------------

ARM_ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")


@cache
def arm_correlations() -> tuple[list[str], np.ndarray, np.ndarray]:
    train = pd.read_csv(ROOT / TRAIN)
    columns = [f"{iso}_pIC50_{arm}" for arm in ("direct_inhibition", "TDI_condition") for iso in ARM_ISOFORMS]
    labels = [f"{iso}, {'direct' if arm == 'direct_inhibition' else 'TDI arm'}" for arm in ("direct_inhibition", "TDI_condition") for iso in ARM_ISOFORMS]
    matrix = np.full((len(columns), len(columns)), np.nan)
    counts = np.zeros_like(matrix)
    for i, a in enumerate(columns):
        for j, b in enumerate(columns):
            if a == b:
                counts[i, j], matrix[i, j] = train[a].notna().sum(), 1.0
                continue
            both = train[[a, b]].dropna()
            counts[i, j] = len(both)
            matrix[i, j] = spearmanr(both[a], both[b])[0] if len(both) > 30 else np.nan
    return labels, matrix, counts


def arm_caption() -> str:
    _labels, matrix, _counts = arm_correlations()
    n = len(ARM_ISOFORMS)
    same = [matrix[i, i + n] for i in range(n)]
    others = [matrix[i, j] for i in range(n) for j in range(i + 1, n) if ARM_ISOFORMS[i] != "CYP2D6" and ARM_ISOFORMS[j] != "CYP2D6"]
    two_d6 = [abs(matrix[ARM_ISOFORMS.index("CYP2D6"), j]) for j in range(n) if ARM_ISOFORMS[j] != "CYP2D6"]
    return ("The correlation (Spearman) between the pIC50 of the assay arms of the four enzymes, over the training compounds that have both values. The two arms of one enzyme (ringed) correlate at "
            f"{min(same):.2f} to {max(same):.2f}: a compound that inhibits an enzyme directly almost always inhibits it after pre-incubation too, which is why the label is a small difference of two large and correlated numbers. "
            f"CYP1A2, CYP2C9 and CYP3A4 share inhibitors (the direct arms correlate at {min(others):.2f} to {max(others):.2f}), while CYP2D6 correlates with none of them (at most {max(two_d6):.2f} in absolute value). "
            "That is the basis of the cross-isoform transfer of section 9.7, which did not pass its gate, and the reason the transfer could not help CYP2D6, the enzyme that needed it.")


@figure("data_arm_correlations", "data", "How the assay arms of the four enzymes correlate", arm_caption, data=(TRAIN,), section="9.7")
def arm_heatmap(theme_name):
    th = apply(theme_name)
    labels, matrix, _counts = arm_correlations()
    n = len(ARM_ISOFORMS)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if matrix[i, j] < 0.6 else "black")
    for i in range(n):
        for a, b in ((i, i + n), (i + n, i)):
            ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, edgecolor=th["gold"], linewidth=2))
    fig.colorbar(image, ax=ax, fraction=0.046, label="Spearman correlation")
    fig.tight_layout()
    return fig


# -- the label as a Gaussian -------------------------------------------------------------------------------------------------------------------------

@cache
def label_gaussians() -> pd.DataFrame:
    from cypmode.challenge.label_confidence import (
        ASSAY_FLOOR,
        STD_SUFFIX,
        direct_column,
        tdi_column,
    )

    train = pd.read_csv(ROOT / TRAIN)
    iso = "CYP3A4"
    t, d = train[tdi_column(iso)], train[direct_column(iso)]
    ok = t.notna() & d.notna()
    t, d = t[ok], d[ok]
    st, sd = train.loc[ok, tdi_column(iso) + STD_SUFFIX], train.loc[ok, direct_column(iso) + STD_SUFFIX]
    s = t - np.maximum(d, ASSAY_FLOOR)
    sigma = np.sqrt(st**2 + (d >= ASSAY_FLOOR) * sd**2)
    z = (s - LOG2) / sigma
    frame = pd.DataFrame({"s": s, "sigma": sigma, "z": z}).dropna()
    picks = []
    for target in (0.0, 1.0, 1.96, 3.5):
        picks.append(frame.iloc[int((frame["z"].abs() - target).abs().argmin())])
    out = pd.DataFrame(picks).reset_index(drop=True)
    out["p_positive"] = 1 - norm.cdf((LOG2 - out["s"]) / out["sigma"])
    return out


def gaussian_caption() -> str:
    table = label_gaussians()
    parts = [f"|z| = {abs(r.z):.2f}: shift {r.s:+.2f} with a standard deviation of {r.sigma:.2f}, P(positive) = {r.p_positive:.2f}" for r in table.itertuples()]
    return ("The label as a threshold of two noisy numbers, drawn for four real CYP3A4 training compounds chosen by how far their shift is from the threshold. Each curve is the distribution of the shift s = t - max(d, 4) of one compound (the Gaussian with the mean and the standard deviation that its two assay arms give), "
            "the dashed line is the threshold log10 2, and the shaded area to its right is the probability that the compound is a time-dependent inhibitor. " + "; ".join(parts) + ". "
            "A compound whose shift sits within a standard deviation of the threshold is nearly a coin flip, and the official score almost certainly leaves such compounds out; the confidence z = (s - log10 2) / sd(s) is how the project decides which compounds count.")


@figure("data_label_gaussian", "data", "The label as a Gaussian around a threshold", gaussian_caption, data=(TRAIN,), section="9.3")
def label_gaussian(theme_name):
    th = apply(theme_name)
    table = label_gaussians()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=False)
    for ax, row in zip(axes, table.itertuples(), strict=True):
        x = np.linspace(row.s - 4 * row.sigma, row.s + 4 * row.sigma, 300)
        y = norm.pdf(x, row.s, row.sigma)
        ax.plot(x, y, color=th["blue"], lw=2)
        ax.fill_between(x, 0, y, where=x > LOG2, color=th["red"], alpha=0.45)
        ax.axvline(LOG2, color=th["text"], ls="--", lw=1.2)
        ax.set(title=f"|z| = {abs(row.z):.2f}: P(positive) = {row.p_positive:.2f}", xlabel="shift s")
        ax.set_yticks([])
        tidy(ax, th, grid=False)
    fig.tight_layout()
    return fig


# -- the three structure engines ---------------------------------------------------------------------------------------------------------------------

@cache
def engine_table() -> pd.DataFrame:
    frames = [pd.read_csv(ROOT / OUT / name) for name in ("benchmark_engine_pose_validation.csv", "benchmark_engine_pose_validation_bioemu2.csv")]
    table = pd.concat(frames, ignore_index=True)
    table = table[table["compound"].str.startswith("bench-") & ~table["compound"].str.contains("-febond|-fecontact")]
    return table


def engine_caption() -> str:
    table = engine_table()
    parts = []
    for engine in ("boltz", "openfold3", "bioemu2"):
        part = table[table["engine"] == engine]
        first = part.sort_values("sample").groupby("compound").first()["true_rmsd"]
        parts.append(f"{engine}: {int((first <= 2).sum())} of {len(first)} ligands with the first sample within 2 A, the best of the pool within 2 A for {int((part.groupby('compound')['true_rmsd'].min() <= 2).sum())}")
    return ("The three structure engines on the same crystal-anchored benchmark ligands: Boltz-2 (the engine of the recipe), OpenFold3 and BioEmu2 (a diffusion model of conformational ensembles that runs here with its pure-PyTorch kernel). "
            "One dot per sample of a pool, at its RMSD to the crystal pose; the filled marker is the first sample and the bar the best of the pool. " + "; ".join(parts) + ". "
            "The engine-consensus gate asks whether a second engine helps to choose among the poses of the first; it is NOT MET, and the final recipe uses Boltz-2 only.")


@figure("struct_engine_comparison", "structure", "The three structure engines on the benchmark ligands", engine_caption, data=(f"{OUT}/benchmark_engine_pose_validation.csv", f"{OUT}/benchmark_engine_pose_validation_bioemu2.csv"), section="11.9")
def engine_comparison(theme_name):
    th = apply(theme_name)
    table = engine_table()
    ligands = list(table.groupby("compound")["true_rmsd"].min().sort_values().index)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for k, (engine, color) in enumerate((("boltz", th["blue"]), ("openfold3", th["orange"]), ("bioemu2", th["teal"]))):
        part = table[table["engine"] == engine]
        for position, ligand in enumerate(ligands):
            rows = part[part["compound"] == ligand].sort_values("sample")
            if rows.empty:
                continue
            x = position + (k - 1) * 0.24
            ax.vlines(x, rows["true_rmsd"].min(), rows["true_rmsd"].max(), color=color, lw=3, alpha=0.4)
            ax.scatter(np.full(len(rows), x), rows["true_rmsd"], s=14, color=color, alpha=0.8)
            ax.scatter(x, rows["true_rmsd"].iloc[0], s=60, color=color, edgecolor=th["text"], linewidth=1.2, zorder=3)
    ax.axhline(2.0, color=th["muted"], ls=":", lw=1.2)
    ax.set_xticks(range(len(ligands)), [c.removeprefix("bench-") for c in ligands], rotation=40, fontsize=8)
    ax.set(ylabel="RMSD to the crystal pose (Å)", title="one dot per sample; the ringed dot is the first sample")
    for engine, color in (("Boltz-2", th["blue"]), ("OpenFold3", th["orange"]), ("BioEmu2", th["teal"])):
        ax.scatter([], [], color=color, label=engine)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- the composition of the pool of the final recipe (held back) -------------------------------------------------------------------------------------

RUN_KINDS = {"": "base run", "-u": "unguided run", "-rs": "restrained run"}


def run_kind(run: str) -> str:
    suffix = re.sub(r"^OCNT-\d+", "", run)
    return RUN_KINDS.get(suffix, "extra samples")


@cache
def recipe_pool() -> tuple[pd.DataFrame, dict]:
    import json

    pools = pd.read_csv(ROOT / OUT / "final/official_posebusters_pools.csv")
    pools["kind"] = pools["run"].map(run_kind)
    chosen = {lig["compound"]: run_kind(lig["run"]) for lig in json.loads((ROOT / OUT / "final/manifest.json").read_text())["ligands"]}
    return pools, chosen


def recipe_caption() -> str:
    _pools, chosen = recipe_pool()
    kinds = Counter(chosen.values())
    return ("What the pool of each of the 20 official ligands is made of (stacked bars: the base run, the unguided run, the restrained run that carries the iron-nitrogen restraint, and the extra samples) and which kind of run the final pose came from (the ringed segment). "
            "The poses are chosen by the recipe: among the valid poses with a margin of at least 0.8, the model's own ranking, the restrained run being read first for a ligand with a Type II donor. The final poses come from the " + ", ".join(f"{k} for {n}" for k, n in kinds.most_common()) + " ligands. "
            "This figure shows predictions for the blinded compounds and is held back until the challenge closes.")


@figure("struct_official20_recipe", "structure", "The pool of each official ligand and the run of its final pose (held back)", recipe_caption, data=(f"{OUT}/final/official_posebusters_pools.csv", f"{OUT}/final/manifest.json"), section="11.10")
def official_recipe(theme_name):
    th = apply(theme_name)
    pools, chosen = recipe_pool()
    order = list(chosen)
    kinds = ["base run", "unguided run", "restrained run", "extra samples"]
    colors = {"base run": th["muted"], "unguided run": th["blue"], "restrained run": th["green"], "extra samples": th["orange"]}
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for row, ligand in enumerate(order):
        left = 0
        for kind in kinds:
            n = int(((pools["ligand"] == ligand) & (pools["kind"] == kind)).sum())
            ax.barh(row, n, left=left, color=colors[kind], edgecolor=th["text"] if chosen[ligand] == kind else "none", linewidth=2.2)
            left += n
    ax.set_yticks(range(len(order)), order, fontsize=7.5)
    ax.invert_yaxis()
    ax.set(xlabel="poses in the pool", title="the pool of each ligand; the ringed segment holds the final pose")
    ax.legend(handles=[Patch(color=colors[k], label=k) for k in kinds], frameon=False, fontsize=8, loc="lower right")
    tidy(ax, th, grid=False)
    fig.tight_layout()
    return fig


# -- the map of the repository ---------------------------------------------------------------------------------------------------------------------

@cache
def repository_files() -> pd.DataFrame:
    names = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    rows = []
    for name in names:
        path = ROOT / name
        if path.is_file():
            rows.append({"folder": name.split("/", 1)[0] if "/" in name else "(root)", "bytes": path.stat().st_size, "python": name.endswith(".py")})
    return pd.DataFrame(rows)


def map_caption() -> str:
    files = repository_files()
    by = files.groupby("folder").agg(files=("bytes", "size"), megabytes=("bytes", lambda b: b.sum() / 1e6))
    top = by.sort_values("files", ascending=False).head(3)
    return (f"The repository by top-level folder: the number of files that git tracks ({len(files)} in all, {files['bytes'].sum() / 1e6:.0f} MB) and their size. The three folders with the most files are "
            + ", ".join(f"{name} ({int(r.files)})" for name, r in top.iterrows()) + ". The code is in `cypmode/` (the engines of the two tracks and the audit), the tests in `tests/`, the results and caches that are small enough for git in `data/`, "
            "the notebook in `notebooks/`, the figures, the card and the film in `docs/`, and the queue that runs the experiments in `queue/`; what is too large or must wait until the deadline is listed in `.gitignore`.")


@figure("eng_repository_map", "engineering", "The repository by folder", map_caption, data=("git ls-files",), section="12.5")
def repository_map(theme_name):
    th = apply(theme_name)
    files = repository_files()
    by = files.groupby("folder").agg(files=("bytes", "size"), megabytes=("bytes", lambda b: b.sum() / 1e6)).sort_values("files")
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(by) + 1.6), sharey=True)
    axes[0].barh(by.index, by["files"], color=th["blue"])
    axes[0].set(xlabel="tracked files", xscale="log", title="files")
    axes[1].barh(by.index, by["megabytes"], color=th["orange"])
    axes[1].set(xlabel="megabytes", xscale="log", title="size")
    for ax in axes:
        tidy(ax, th)
    fig.tight_layout()
    return fig


# -- the flow of the data and the packages of the code (the two diagrams of the project map, drawn as figures) --------------------------------------------

DATAFLOW_TITLE = "The flow of the data, from the inputs to the submission files"
CODE_TITLE = "The packages of the code; an arrow goes from a package to a package whose modules it imports, and its label counts those imports"


def _node(ax, centre, size, title, detail, theme, accent):
    """A rounded box with a bold title and a muted line of detail; returns the patch, so that an arrow can start and end on its border."""
    (x, y), (w, h) = centre, size
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.16", linewidth=2.0, edgecolor=accent, facecolor=theme["panel"], zorder=3)
    ax.add_patch(patch)
    if detail:
        ax.text(x, y + h * 0.17, title, ha="center", va="center", fontsize=13, fontweight="bold", color=theme["text"], zorder=4)
        ax.text(x, y - h * 0.22, detail, ha="center", va="center", fontsize=10.5, color=theme["muted"], zorder=4)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=13, fontweight="bold", color=theme["text"], zorder=4)
    return patch


def _inside(patch, point) -> bool:
    x0, y0, w, h = patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()
    return x0 <= point[0] <= x0 + w and y0 <= point[1] <= y0 + h


def _link(ax, a, b, theme, color, label=None, rad=0.0, width=2.2, at=0.5, dy=0.0, shift_end=(0.0, 0.0), shift_start=(0.0, 0.0)):
    """An arrow from the patch ``a`` to the patch ``b``, clipped on their borders; the label sits on the visible part of the curve (``at`` is the share of it)."""
    start = (a.get_x() + a.get_width() / 2 + shift_start[0], a.get_y() + a.get_height() / 2 + shift_start[1])
    end = (b.get_x() + b.get_width() / 2 + shift_end[0], b.get_y() + b.get_height() / 2 + shift_end[1])
    ax.add_patch(FancyArrowPatch(start, end, patchA=a, patchB=b, arrowstyle="-|>", mutation_scale=18, linewidth=width, color=color, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=2, zorder=2))
    if label is None:
        return
    control = ((start[0] + end[0]) / 2 + rad * (end[1] - start[1]), (start[1] + end[1]) / 2 - rad * (end[0] - start[0]))  # the control point of the arc3 curve
    curve = [((1 - t) ** 2 * start[0] + 2 * t * (1 - t) * control[0] + t * t * end[0], (1 - t) ** 2 * start[1] + 2 * t * (1 - t) * control[1] + t * t * end[1])
             for t in np.linspace(0, 1, 401)]
    visible = [q for q in curve if not _inside(a, q) and not _inside(b, q)]
    x, y = visible[int(at * (len(visible) - 1))]
    ax.text(x, y + dy, label, ha="center", va="center", fontsize=11.5, fontweight="bold", color=theme["text"], zorder=5,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": theme["background"], "edgecolor": color, "linewidth": 1.1})


def _heading(fig, text, theme, wrap):
    import textwrap

    fig.suptitle("\n".join(textwrap.wrap(text, wrap)), fontsize=15, fontweight="bold", color=theme["text"], y=0.985)


def dataflow_caption() -> str:
    from cypmode.audit import project_map

    facts = project_map.dataflow_facts()
    return (f"{DATAFLOW_TITLE}. The inputs of a track enter the jobs of the queue, and {facts['writers']} of the {len(facts['jobs'])} jobs write a log. The jobs write {len(facts['written'])} log names. The ledger reads "
            f"the lines of the gates in every log, and {len(facts['named'])} logs are also read by name. The reports, the figures, and the notebook are built from the logs and the ledger by the chain 900 to 960, "
            "and the submission files follow from them. The same diagram, written in Mermaid, is `docs/map/dataflow.md`.")


@figure("eng_data_flow", "engineering", "The flow of the data, from the inputs to the submission files", dataflow_caption, data=("queue/jobs", "cypmode/audit/project_map.py"), section="12.5")
def data_flow(theme_name):
    from cypmode.audit import project_map

    th = apply(theme_name)
    facts = project_map.dataflow_facts()
    fig, ax = plt.subplots(figsize=(18, 5.6))
    canvas(ax, (0, 25.2), (-0.3, 7.0))
    ax.set_aspect("auto")
    inputs = _node(ax, (1.5, 3.2), (2.6, 1.9), "inputs", "data/*/inputs", th, th["green"])
    jobs = _node(ax, (5.9, 3.2), (3.4, 1.9), "queue jobs", f"{facts['writers']} of {len(facts['jobs'])} write a log", th, th["blue"])
    logs = _node(ax, (11.3, 3.2), (2.2, 1.9), "logs", "", th, th["orange"])
    ledger = _node(ax, (18.1, 5.6), (4.6, 1.9), "ledger", "cypmode/audit/ledger_kinds.py", th, th["purple"])
    reports = _node(ax, (18.1, 1.0), (4.6, 1.9), "reports, figures, notebook", "built by the chain 900 to 960", th, th["teal"])
    submissions = _node(ax, (23.3, 1.0), (3.0, 1.9), "submission files", "", th, th["red"])
    _link(ax, inputs, jobs, th, th["muted"])
    _link(ax, jobs, logs, th, th["muted"], f"{len(facts['written'])}\nlog names")
    _link(ax, logs, ledger, th, th["muted"], "every log,\nthrough gate_rows", at=0.5)
    _link(ax, logs, reports, th, th["muted"], f"{len(facts['named'])} logs are\nread by name", at=0.5)
    _link(ax, ledger, reports, th, th["muted"])
    _link(ax, reports, submissions, th, th["muted"])
    _heading(fig, DATAFLOW_TITLE, th, 90)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.04)
    return fig


def code_caption() -> str:
    from collections import Counter

    from cypmode.audit import project_map

    facts = project_map.code_facts()
    packages = Counter(project_map.package_of(m) for m in facts["paths"])
    edges = sum(len(t) for t in facts["graph"].values())
    return (f"{CODE_TITLE}. The project has {len(facts['paths'])} modules in {len(packages)} packages, and the modules hold {edges} imports of one another, of which {sum(facts['cross'].values())} cross from one package "
            "to another. The width of an arrow follows its count. The same diagram, written in Mermaid, is `docs/map/code.md`, which also lists the modules of every package.")


CODE_POSITIONS = {"audit": (2.2, 6.0), "explorer": (8.0, 9.0), "post_deadline": (8.0, 3.0), "data": (8.0, 1.0), "challenge": (14.0, 6.0), "validation": (19.6, 6.0),
                  "top level": (24.8, 9.0), "metrics": (24.8, 3.0)}
CODE_COLOURS = {"audit": "blue", "challenge": "orange", "validation": "teal", "explorer": "purple", "post_deadline": "gold", "data": "green", "metrics": "red", "top level": "muted"}
# Every arrow runs along horizontal and vertical lines. The points are the corners of its path, from the border of the box it leaves to the border of the box it enters;
# the last pair gives the label the segment that carries it and the share of that segment where it sits.
CODE_ROUTES = {
    ("audit", "challenge"): ([(3.8, 6.0), (12.4, 6.0)], 0, 0.49),
    ("challenge", "validation"): ([(15.6, 5.7), (18.0, 5.7)], 0, 0.5),
    ("validation", "challenge"): ([(18.0, 6.3), (15.6, 6.3)], 0, 0.5),
    ("challenge", "top level"): ([(14.4, 6.6), (14.4, 8.6), (23.2, 8.6)], 1, 0.3),
    ("challenge", "metrics"): ([(14.8, 5.4), (14.8, 2.2), (22.4, 2.2), (22.4, 2.8), (23.2, 2.8)], 1, 0.5),
    ("audit", "validation"): ([(1.6, 6.6), (1.6, 10.4), (19.0, 10.4), (19.0, 6.6)], 1, 0.4),
    ("audit", "top level"): ([(1.0, 6.6), (1.0, 11.2), (24.8, 11.2), (24.8, 9.6)], 1, 0.5),
    ("audit", "explorer"): ([(2.2, 6.6), (2.2, 9.4), (6.4, 9.4)], 1, 0.5),
    ("explorer", "audit"): ([(6.4, 9.0), (3.2, 9.0), (3.2, 6.6)], 1, 0.5),
    ("audit", "post_deadline"): ([(2.2, 5.4), (2.2, 3.2), (6.4, 3.2)], 1, 0.5),
    ("explorer", "challenge"): ([(9.6, 8.6), (11.0, 8.6), (11.0, 6.4), (12.4, 6.4)], 1, 0.5),
    ("explorer", "top level"): ([(9.6, 9.4), (23.2, 9.4)], 0, 0.3),
    ("post_deadline", "challenge"): ([(9.6, 3.4), (11.4, 3.4), (11.4, 5.65), (12.4, 5.65)], 1, 0.5),
    ("post_deadline", "validation"): ([(9.6, 2.9), (19.2, 2.9), (19.2, 5.4)], 0, 0.75),
    ("data", "challenge"): ([(9.6, 1.2), (13.2, 1.2), (13.2, 5.4)], 1, 0.75),
    ("data", "metrics"): ([(9.6, 0.8), (24.8, 0.8), (24.8, 2.4)], 0, 0.5),
    ("validation", "top level"): ([(21.2, 6.3), (22.2, 6.3), (22.2, 9.0), (23.2, 9.0)], 1, 0.5),
    ("validation", "metrics"): ([(20.4, 5.4), (20.4, 3.3), (23.2, 3.3)], 1, 0.5),
}


def _route(ax, points, theme, width, label, which, share):
    """An arrow along the corners ``points`` (rounded at every corner), an arrow head at the last point and a label on one of its segments."""
    from matplotlib.patches import PathPatch, Polygon
    from matplotlib.path import Path as MplPath

    head, radius = 0.34, 0.28
    pts = [np.array(q, dtype=float) for q in points]
    direction = (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])
    tip = pts[-1]
    pts[-1] = tip - direction * head
    vertices, codes = [tuple(pts[0])], [MplPath.MOVETO]
    for i in range(1, len(pts) - 1):
        before, after = pts[i - 1], pts[i + 1]
        r = min(radius, np.linalg.norm(pts[i] - before) / 2, np.linalg.norm(after - pts[i]) / 2)
        vertices += [tuple(pts[i] - (pts[i] - before) / np.linalg.norm(pts[i] - before) * r), tuple(pts[i]), tuple(pts[i] + (after - pts[i]) / np.linalg.norm(after - pts[i]) * r)]
        codes += [MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3]
    vertices.append(tuple(pts[-1]))
    codes.append(MplPath.LINETO)
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor="none", edgecolor=theme["muted"], linewidth=width, capstyle="butt", joinstyle="round", zorder=2))
    side = np.array([-direction[1], direction[0]]) * 0.15
    ax.add_patch(Polygon([tip, pts[-1] + side, pts[-1] - side], closed=True, facecolor=theme["muted"], edgecolor=theme["muted"], linewidth=0.5, zorder=2))
    start, end = np.array(points[which], dtype=float), np.array(points[which + 1], dtype=float)
    x, y = start + (end - start) * share
    ax.text(x, y, label, ha="center", va="center", fontsize=11.5, fontweight="bold", color=theme["text"], zorder=5,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": theme["background"], "edgecolor": theme["muted"], "linewidth": 1.1})


@figure("eng_code_packages", "engineering", "The packages of the code", code_caption, data=("cypmode/**/*.py", "cypmode/audit/project_map.py"), section="12.5")
def code_packages(theme_name):
    from collections import Counter

    from cypmode.audit import project_map

    th = apply(theme_name)
    facts = project_map.code_facts()
    packages = Counter(project_map.package_of(m) for m in facts["paths"])
    fig, ax = plt.subplots(figsize=(17, 7.8))
    canvas(ax, (0, 27.4), (0, 12.0))
    for name, count in packages.items():
        _node(ax, CODE_POSITIONS[name], (3.2, 1.2), name, f"{count} module{'s' if count != 1 else ''}", th, th[CODE_COLOURS[name]])
    biggest = max(facts["cross"].values())
    assert set(facts["cross"]) == set(CODE_ROUTES), "the routes of the arrows are drawn for exactly the edges that the code has"
    for (a, b), count in sorted(facts["cross"].items(), key=lambda item: -item[1]):
        points, which, share = CODE_ROUTES[(a, b)]
        _route(ax, points, th, 1.5 + 2.8 * (count / biggest) ** 0.6, str(count), which, share)
    _heading(fig, CODE_TITLE, th, 100)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01)
    return fig
