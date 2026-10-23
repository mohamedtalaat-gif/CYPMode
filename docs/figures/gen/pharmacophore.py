# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Figures about the pharmacophore representations of the project: the pockets as density fields in the frame of the heme, and every pharmacophore representation with its verdict.

Both are drawn from the files of the runs: the fields that ``cypmode.challenge.pocket_pharmacophore --extract`` wrote, the logs of the gated experiments, the recorded comparisons of the earlier
blocks and the log of the pose terms. A source that does not exist yet is skipped, so the second figure grows as the queue finishes.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import figure
from style import apply, tidy

GROUP = "pharmacophore"
ROOT = Path(__file__).resolve().parents[3]
LOGS = ROOT / "data/structure_track/outputs/cyp3a4/logs"
COMPARISONS = ROOT / "data/tdi_track/outputs/comparisons"
FIELDS = "data/tdi_track/outputs/caches/pocket_fields_{pocket}.npz"
REFERENCES = {"cyp3a4": "data/structure_track/inputs/cyp3a4_reference_ligands.json", "cyp2d6": "data/structure_track/inputs/cyp2d6_reference_ligands.json"}
PANELS = (("Donor", "Acceptor", "Aromatic", "Hydrophobe", "occupancy"))
STACKED = re.compile(r"z0=([\d.]+)\s+stacked base\+added vs base: AUC ([+-][\d.]+) \(p=([\d.]+)")
POSE_TERM = re.compile(r"^(physics|pharmacophore|combined): mean Spearman rho ([+-][\d.]+), positive for (\d+)% of (\d+) ligands", re.MULTILINE)
GATE_RHO, GATE_SHARE = 0.3, 60


def reference_counts() -> dict[str, tuple[int, int]]:
    """{pocket: (ligand copies, distinct ligands)} of the crystal reference sets."""
    out = {}
    for pocket, path in REFERENCES.items():
        ligands = json.loads((ROOT / path).read_text())["ligands"]
        out[pocket] = (len(ligands), len({r["comp_id"] for r in ligands}))
    return out


def fields_caption() -> str:
    counts = reference_counts()
    return ("The pocket of each enzyme as the public crystal ligands put in the frame of the heme: the iron is at the origin of each panel, the heme plane is the plane of the page and z, the distance above it, "
            "is collapsed by taking the maximum along it. Each column is the density of one feature family of the ligands (donor, acceptor, aromatic ring, hydrophobe) and the last is the occupancy of their heavy "
            f"atoms. CYP3A4 rests on {counts['cyp3a4'][0]} ligand copies of {counts['cyp3a4'][1]} distinct ligands and CYP2D6 on {counts['cyp2d6'][0]} copies of {counts['cyp2d6'][1]}, so the CYP2D6 fields are the "
            "less certain of the two.")


@figure("pharm_pocket_fields", GROUP, "The pockets of CYP3A4 and CYP2D6 as pharmacophore fields in the frame of the heme", fields_caption,
        data=(FIELDS.format(pocket="cyp3a4"), FIELDS.format(pocket="cyp2d6"), REFERENCES["cyp3a4"], REFERENCES["cyp2d6"]), section="9.13")
def pocket_fields(theme_name):
    th = apply(theme_name)
    fig, axes = plt.subplots(2, len(PANELS), figsize=(11.0, 4.6), constrained_layout=True)
    extent = (-14.5, 14.5, -14.5, 14.5)
    for row, pocket in enumerate(("cyp3a4", "cyp2d6")):
        with np.load(ROOT / FIELDS.format(pocket=pocket)) as archive:
            fields = {name: archive[name] for name in archive.files}
        for column, name in enumerate(PANELS):
            ax = axes[row][column]
            image = fields[name].max(axis=2).T  # the maximum along z, drawn with x to the right and y up
            ax.imshow(image, origin="lower", extent=extent, cmap="viridis" if theme_name == "dark" else "YlGnBu", vmin=0, vmax=1)
            ax.plot([0], [0], marker="o", color=th["orange"], markersize=6, markeredgecolor=th["edge"])
            ax.set_xticks([-10, 0, 10])
            ax.set_yticks([-10, 0, 10])
            if row == 0:
                ax.set_title(name if name != "occupancy" else "heavy-atom occupancy", fontsize=10)
            if column == 0:
                ax.set_ylabel(f"{pocket.upper()}\ny (Å)", fontsize=10)
            if row == 1:
                ax.set_xlabel("x (Å)", fontsize=10)
    return fig


def stacked_rows() -> list[dict]:
    """Rows for the panel of AUC changes: one per representation and enzyme, from the stacked comparison of its log (two cuts)."""
    sources = (
        ("physics block, 4 levels", "physics_experiment_{iso}.log", ("CYP3A4", "CYP2D6")),
        ("physics block, 20 columns", "physics_compact_{iso}.log", ("CYP3A4", "CYP2D6")),
        ("physics block, tier B", "physics_tierb_{iso}.log", ("CYP3A4", "CYP2D6")),
        ("pocket block, own pocket", "aux_pocket_{pocket}_{iso}.log", ("CYP3A4", "CYP2D6")),
        ("pocket block, other pocket (control)", "exploratory_pocket_control_{iso}.log", ("CYP3A4", "CYP2D6")),
    )
    rows = []
    for label, pattern, isoforms in sources:
        for iso in isoforms:
            name = pattern.format(iso=iso, pocket=iso.lower())
            path = LOGS / name
            if not path.exists():
                continue
            cuts = {float(z0): (float(gain), float(p)) for z0, gain, p in STACKED.findall(path.read_text(errors="ignore"))}
            if cuts:
                rows.append({"label": label, "iso": iso, "cuts": cuts})
    return rows


def older_rows() -> list[dict]:
    """Rows for the panel of MCC changes: the 3-point fingerprint and the quantum-pharmacophore block against the same base, from the recorded paired comparisons."""
    out = []
    for label, name, variant in (("3-point pharmacophore fingerprint", "pharmacophore_fingerprint_significance_paired.csv", "M3+Pharmacophore"),
                                 ("quantum-pharmacophore block", "quantum_pharmacophore_significance_paired.csv", "M3+Quantum")):
        path = COMPARISONS / name
        if not path.exists():
            continue
        table = pd.read_csv(path)
        for iso, group in table.groupby("isoform"):
            pick = group[(group["group1"] == "M3") & (group["group2"] == variant)]
            if len(pick):
                row = pick.iloc[0]
                out.append({"label": label, "iso": iso, "gain": float(row["meandiff"]), "p": float(row["p-adj"]), "low": float(row["lower"]), "high": float(row["upper"])})
    return out


def pose_rows() -> list[dict]:
    path = LOGS / "pose_awareness.log"
    if not path.exists():
        return []
    return [{"term": term, "rho": float(rho), "share": int(share), "n": int(n)} for term, rho, share, n in POSE_TERM.findall(path.read_text(errors="ignore"))]


def forest_caption() -> str:
    stacked, older, poses = stacked_rows(), older_rows(), pose_rows()
    passing = [r for r in stacked if r["label"] != "pocket block, other pocket (control)" and all(p < 0.05 and g > 0 for g, p in r["cuts"].values()) and len(r["cuts"]) == 2]
    return (f"Every pharmacophore representation of the project and what happened to it (the fourth level of the physics block is its pharmacophore level). Left: the change in AUC when the block is added to the default stack, at the two confidence cuts, for the "
            f"{len(stacked)} comparisons that have a log (a point is filled when its corrected p is below 0.05); the gate asks for both cuts, and {len(passing)} of these comparisons pass it. Middle: the change in MCC "
            f"of the older blocks against the same base, with the 95% interval and the Holm-adjusted p ({len(older)} comparisons). Right: the Spearman correlation of the pose terms with minus the true RMSD "
            f"over the benchmark poses, with the gate at {GATE_RHO} ({len(poses)} terms).")


@figure("pharm_forest", GROUP, "Every pharmacophore representation of the project and its verdict", forest_caption,
        data=("data/structure_track/outputs/cyp3a4/logs/physics_experiment_CYP3A4.log", "data/tdi_track/outputs/comparisons/pharmacophore_fingerprint_significance_paired.csv",
              "data/structure_track/outputs/cyp3a4/logs/pose_awareness.log"), section="9.13")
def forest(theme_name):
    th = apply(theme_name)
    stacked, older, poses = stacked_rows(), older_rows(), pose_rows()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, max(4.4, 0.5 * len(stacked) + 1.8)), gridspec_kw={"width_ratios": [3.0, 2.6, 1.8]}, constrained_layout=True)
    ax = axes[0]
    for i, row in enumerate(stacked):
        for k, (z0, (gain, p)) in enumerate(sorted(row["cuts"].items())):
            colour = th["green"] if p < 0.05 and gain > 0 else th["muted"]
            ax.plot(gain, i + (k - 0.5) * 0.28, "o", color=colour, markerfacecolor=colour if p < 0.05 else "none", markersize=6)
            ax.text(gain + 0.0004, i + (k - 0.5) * 0.28, f"p={p:.2f}", va="center", fontsize=7, color=th["muted"])
    ax.axvline(0, color=th["muted"], linewidth=0.8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_yticks(range(len(stacked)))
    ax.set_yticklabels([f"{r['label']} · {r['iso']}" for r in stacked], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("change in AUC when added to the stack\n(two cuts per row)", fontsize=9)
    tidy(ax, th)
    ax = axes[1]
    for i, row in enumerate(older):
        colour = th["green"] if row["p"] < 0.05 and row["gain"] > 0 else th["muted"]
        ax.errorbar(row["gain"], i, xerr=[[row["gain"] - row["low"]], [row["high"] - row["gain"]]], fmt="o", color=colour, markersize=6, capsize=3)
        ax.text(row["high"] + 0.004, i, f"p={row['p']:.2f}", va="center", fontsize=7, color=th["muted"])
    ax.axvline(0, color=th["muted"], linewidth=0.8)
    ax.set_yticks(range(len(older)))
    ax.set_yticklabels([f"{r['label']} · {r['iso']}" for r in older], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("change in MCC (95% interval,\nHolm-adjusted p)", fontsize=9)
    tidy(ax, th)
    ax = axes[2]
    for i, row in enumerate(poses):
        passes = row["rho"] > GATE_RHO and row["share"] >= GATE_SHARE
        ax.barh(i, row["rho"], color=th["green"] if passes else th["muted"])
        ax.text(max(row["rho"], 0) + 0.01, i, f"{row['share']}% of {row['n']}", va="center", fontsize=7, color=th["muted"])
    ax.axvline(GATE_RHO, color=th["red"], linestyle="--", linewidth=1)
    ax.set_yticks(range(len(poses)))
    ax.set_yticklabels([f"pose term: {r['term']}" for r in poses], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.1, 0.6)
    ax.set_xlabel("Spearman ρ with −RMSD\n(gate: ρ > 0.3)", fontsize=9)
    tidy(ax, th)
    return fig
