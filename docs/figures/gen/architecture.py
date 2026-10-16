"""Block diagrams of the two tracks and the nested cross-validation grid; every count in them is read from the project's files."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from catalog import figure
from cypmode.audit.method_report import gate_rows
from gen._draw import arrow, box, canvas
from style import apply

GROUP = "architecture"
ROOT = Path(__file__).resolve().parents[3]
TRAIN = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"
NESTED = "data/tdi_track/outputs/nested"
OFFICIAL_POSEBUSTERS = "data/structure_track/outputs/cyp3a4/posebusters_validity_official20.csv"


def tdi_facts() -> dict:
    store = np.load(ROOT / STORE, allow_pickle=True)
    blocks = [name for name in store.files if name != "smiles"]
    from cypmode.challenge.nested import NestedPlan
    from cypmode.challenge.nested_runners import default_candidates

    families = Counter(c.name.split("_")[0] for c in default_candidates())
    plan = NestedPlan.load(ROOT / NESTED / "CYP3A4/plan.npz")
    verdicts = [v for _, name, v in gate_rows() if name.startswith("CYP")]  # the gates of the TDI track; the structure gates are drawn in the other pipeline
    return {
        "n_train": len(pd.read_csv(ROOT / TRAIN, usecols=["Molecule_Name"])),
        "n_test": len(pd.read_csv(ROOT / TEST, usecols=["SMILES"])),
        "blocks": len(blocks),
        "columns": int(sum(store[b].shape[1] for b in blocks)),
        "families": families,
        "n_models": sum(families.values()),
        "outer": len(plan.outer_folds()),
        "outer_splits": plan.outer_splits,
        "inner_splits": plan.inner_splits,
        "gates": len(verdicts),
        "gates_met": verdicts.count("MET"),
    }


def tdi_caption() -> str:
    f = tdi_facts()
    return (f"The TDI track from the training file to the submitted calls. {f['n_train']} training compounds with two assay arms become a label through the shift rule; "
            f"the feature store holds {f['columns']} columns in {f['blocks']} blocks, from which {f['n_models']} default candidate models draw; every comparison is made on held-out outer folds ({f['outer']} of them) that no choice has seen; "
            f"a stacked logistic blend combines the candidates; {f['gates']} pre-declared TDI gates decided which further ideas were allowed in ({f['gates_met']} met).")


@figure("arch_tdi_pipeline", GROUP, "The TDI track as a pipeline", tdi_caption, data=(TRAIN, TEST, STORE, NESTED), section="0.2")
def tdi_pipeline(theme_name):
    th = apply(theme_name)
    f = tdi_facts()
    fam = ", ".join(f"{n} {k}" for k, n in f["families"].most_common())
    steps = [
        ("Training file", f"{f['n_train']} compounds, two assay arms per isoform", th["panel"]),
        ("Label rule", "s = t - max(d, 4) > log10 2, with a confidence z from the arms' standard deviations", th["panel"]),
        ("Features", f"feature store: {f['columns']} columns in {f['blocks']} blocks (fingerprints, descriptors, alerts, physics)", th["panel"]),
        ("Candidate models", f"{f['n_models']}: {fam}", th["panel"]),
        ("Nested validation", f"{f['outer']} outer folds that no choice sees; weights match the test set", th["orange"]),
        ("Stacked blend", "logistic regression over the candidates; its penalty is chosen inside the inner folds", th["panel"]),
        ("Gates", f"each new idea must beat the stack with corrected p < 0.05 at both confidence cuts: {f['gates']} verdicts, {f['gates_met']} met", th["orange"]),
        ("Submission", f"{f['n_test']} test compounds scored; top fraction called; portal check; CSV", th["green"]),
    ]
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    canvas(ax, (0, 12.4), (0, 5.6))
    xs = [1.6, 4.6, 7.6, 10.6]
    for i, (title, text, fill) in enumerate(steps):
        x = xs[i % 4] if i < 4 else xs[3 - i % 4]
        y = 4.1 if i < 4 else 1.5
        box(ax, x, y, 2.7, 2.0, text, fill, th["text"], "#1A1A1A" if fill in (th["orange"], th["green"]) else th["text"], size=8.3, wrap=27, title=title)
        if i not in (3, 7):
            nxt = xs[(i + 1) % 4] if i + 1 < 4 else xs[3 - (i + 1) % 4]
            arrow(ax, (x + (1.4 if i < 3 else -1.4), y), (nxt + (-1.4 if i < 3 else 1.4), y), th["text"])
    arrow(ax, (xs[3], 3.05), (xs[3], 2.55), th["text"])
    fig.tight_layout()
    return fig


def structure_facts() -> dict:
    poses = pd.read_csv(ROOT / OFFICIAL_POSEBUSTERS)
    return {"poses": len(poses), "failures": int((~poses["all_passed"].astype(bool)).sum())}


def structure_caption() -> str:
    f = structure_facts()
    return (f"The structure track from a ligand to the submitted zip. Boltz-2 folds each ligand into CYP3A4 with the heme; when the ligand has a nitrogen that can reach the iron, a bond-length restraint "
            f"is part of the input and its run is read first; the first pose that passes the validity checks is taken. OpenFold3 was run on the same benchmark and did not improve the poses, "
            f"so it is not in the final recipe (dashed). The zip holds {f['poses']} poses, of which {f['failures']} fail any PoseBusters check, and its bytes are the same every time it is rebuilt.")


@figure("arch_structure_pipeline", GROUP, "The structure track as a pipeline", structure_caption, data=(OFFICIAL_POSEBUSTERS,), section="11.1")
def structure_pipeline(theme_name):
    th = apply(theme_name)
    f = structure_facts()
    steps = [
        ("Target", "CYP3A4 with the heme; cryo-EM templates were tried", th["panel"]),
        ("Input per ligand", "SMILES, heme component, sequence, and the Fe-N bond restraint for ligands with a donor nitrogen", th["panel"]),
        ("Boltz-2", "co-folding on the Apple GPU, several samples per ligand", th["blue"]),
        ("Validity", "PoseBusters on every pose; invalid poses are never handed in", th["orange"]),
        ("Selection", "first valid pose, the restrained run first for donor ligands", th["panel"]),
        ("Final zip", f"{f['poses']} PDB files, {f['failures']} PoseBusters failures, identical bytes on rebuild", th["green"]),
    ]
    fig, ax = plt.subplots(figsize=(12.4, 4.6))
    canvas(ax, (0, 12.4), (0, 4.6))
    xs = [1.1 + 2.04 * i for i in range(6)]
    for x, (title, text, fill) in zip(xs, steps, strict=True):
        box(ax, x, 2.8, 1.9, 2.1, text, fill, th["text"], "#FFFFFF" if fill == th["blue"] else ("#1A1A1A" if fill in (th["orange"], th["green"]) else th["text"]), size=7.8, wrap=20, title=title)
    for a, b in zip(xs, xs[1:], strict=False):
        arrow(ax, (a + 0.98, 2.8), (b - 0.98, 2.8), th["text"])
    box(ax, xs[2], 0.75, 2.4, 0.9, "OpenFold3: benchmarked, did not help", th["panel"], th["muted"], th["muted"], size=8, wrap=26)
    arrow(ax, (xs[2], 1.76), (xs[2], 1.22), th["muted"], ls="--")
    fig.tight_layout()
    return fig


def nested_caption() -> str:
    f = tdi_facts()
    return (f"Nested cross-validation as a grid. The compounds are cut into {f['outer_splits']} outer folds; each outer fold in turn is held out and never used for any choice. "
            f"The other {f['outer_splits'] - 1} are cut again into {f['inner_splits']} inner folds that give out-of-fold predictions on which the models are compared, the blend is fitted and the "
            f"call threshold is set. The whole procedure is repeated with new random cuts ({f['outer']} outer folds in all with the repeats), and the held-out scores are what is reported.")


@figure("arch_nested_cv", GROUP, "Nested cross-validation as a picture", nested_caption, data=(NESTED,), section="6.2")
def nested_cv(theme_name):
    th = apply(theme_name)
    f = tdi_facts()
    outer, inner = f["outer_splits"], f["inner_splits"]
    fig, ax = plt.subplots(figsize=(10.5, 0.8 * outer + 2.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.4, outer + 0.9)
    ax.invert_yaxis()
    ax.axis("off")
    width = 9.0 / outer
    for row in range(outer):
        for col in range(outer):
            held = col == row
            ax.add_patch(Rectangle((1.6 + col * width, row + 0.1), width * 0.94, 0.78, facecolor=th["red"] if held else th["panel"], edgecolor=th["text"], lw=1.2))
            if not held and row == 0:
                for k in range(inner):
                    ax.add_patch(Rectangle((1.6 + col * width + k * width * 0.94 / inner, row + 0.1), width * 0.94 / inner, 0.78, facecolor="none", edgecolor=th["blue"], lw=0.9))
        ax.text(1.4, row + 0.5, f"outer fold {row + 1}", ha="right", va="center", fontsize=9, color=th["text"])
    ax.text(1.6, -0.15, "compounds", fontsize=9, color=th["muted"], va="bottom")
    ax.add_patch(Rectangle((1.6, outer + 0.2), 0.4, 0.3, facecolor=th["red"], edgecolor=th["text"]))
    ax.text(2.15, outer + 0.35, "held out: scores reported on these only", fontsize=9, va="center", color=th["text"])
    ax.add_patch(Rectangle((6.6, outer + 0.2), 0.4, 0.3, facecolor=th["panel"], edgecolor=th["blue"], lw=1.4))
    ax.text(7.15, outer + 0.35, f"first row: the other folds are cut into {inner} inner folds", fontsize=9, va="center", color=th["text"])
    return fig
