# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Figures about the pharmacophore representations and models of the project: the pockets as density fields in the frame of the heme, the sites of the structure-based model, the strongest hypotheses of
the ligand-based model, and every pharmacophore representation and model with its verdict.

All are drawn from the files of the runs: the fields that ``cypmode.challenge.pocket_pharmacophore --extract`` wrote, the hypotheses that ``cypmode.challenge.pharmacophore_hypotheses --extract`` wrote, the
logs of the gated experiments, the recorded comparisons of the earlier blocks and the log of the pose terms. A log that does not exist yet is skipped, so the forest plot grows as the queue finishes.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import figure
from matplotlib.patches import Circle
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


FAMILY_COLOUR = {"Donor": "blue", "Acceptor": "red", "Aromatic": "green", "Hydrophobe": "orange", "PosIonizable": "purple", "NegIonizable": "teal"}
FAMILY_MARK = {"Donor": "D", "Acceptor": "A", "Aromatic": "R", "Hydrophobe": "H", "PosIonizable": "+", "NegIonizable": "\u2212"}


def pocket_fields_of(pocket: str) -> dict[str, np.ndarray]:
    with np.load(ROOT / FIELDS.format(pocket=pocket)) as archive:
        return {name: archive[name] for name in archive.files}


def sites_caption() -> str:
    from cypmode.challenge import pocket_pharmacophore as pp

    counts = {pocket: sum(len(v) for v in pp.sites_of(pocket_fields_of(pocket)).values()) for pocket in pp.POCKETS}
    return (f"The sites of the structure-based pharmacophore model of each pocket, in the frame of the heme (the iron is the orange dot). A site is a local maximum of one of the six feature fields of the crystal ligands "
            f"(value at least {pp.SITE_MIN_VALUE}, at most {pp.SITES_PER_FAMILY} per family, {pp.SITE_SEPARATION} A apart); the circle is its tolerance sphere of {pp.SITE_RADIUS} A, the letter its family "
            f"(D donor, A acceptor, R aromatic ring, H hydrophobe, + and \u2212 ionisable) and the opacity its weight. Grey: the heavy-atom occupancy of the same ligands. Left: seen from above the heme (x, y); "
            f"right: from the side (x, z). CYP3A4 has {counts['cyp3a4']} sites and CYP2D6 {counts['cyp2d6']}.")


@figure("pharm_site_model", GROUP, "The sites of the structure-based pharmacophore model of each pocket", sites_caption,
        data=(FIELDS.format(pocket="cyp3a4"), FIELDS.format(pocket="cyp2d6")), section="9.14")
def site_model(theme_name):
    from cypmode.challenge import pocket_pharmacophore as pp

    th = apply(theme_name)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.6), constrained_layout=True)
    for row, pocket in enumerate(("cyp3a4", "cyp2d6")):
        fields = pocket_fields_of(pocket)
        sites = pp.sites_of(fields)
        views = (("top view (x, y)", fields["occupancy"].max(axis=2), (0, 1), (-14.5, 14.5, -14.5, 14.5)), ("side view (x, z)", fields["occupancy"].max(axis=1), (0, 2), (-14.5, 14.5, -3.5, 16.5)))
        for column, (title, image, (i, j), extent) in enumerate(views):
            ax = axes[row][column]
            ax.imshow(image.T, origin="lower", extent=extent, cmap="Greys" if theme_name == "light" else "Greys_r", vmin=0, vmax=float(fields["occupancy"].max()), alpha=0.75)
            ax.plot([0], [0], marker="o", color=th["orange"], markersize=7, markeredgecolor=th["edge"], zorder=5)
            for family, rows in sites.items():
                for site in rows:
                    ax.add_patch(Circle((site[i], site[j]), pp.SITE_RADIUS, facecolor=th[FAMILY_COLOUR[family]], alpha=0.12 + 0.3 * site[3], edgecolor=th[FAMILY_COLOUR[family]], linewidth=1.6, zorder=3))
                    ax.text(site[i], site[j], FAMILY_MARK[family], ha="center", va="center", fontsize=9, fontweight="bold", color=th[FAMILY_COLOUR[family]], zorder=6)
            ax.set_aspect("equal")
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel("x (\u00c5)", fontsize=9)
            ax.set_ylabel(("y" if column == 0 else "z") + " (\u00c5)", fontsize=9)
            ax.set_title(f"{pocket.upper()}, {title}", fontsize=10)
            tidy(ax, th, grid=False)
    return fig


def hypotheses_caption() -> str:
    from cypmode.challenge import pharmacophore_hypotheses as ph

    return (f"The ligand-based hypothesis model fitted on all the training rows of each enzyme, to show what it holds (the model inside the validation is refitted in every training split, and this fit is "
            f"not a measure of performance). Each bar is one hypothesis: the feature types and the distances between them, as found in one compound that realises it. Green bars are the strongest of the "
            f"{ph.KEEP_PER_SIDE} hypotheses that favour a positive label (the pharmacophore of the positives), red bars the strongest of the {ph.KEEP_PER_SIDE} exclusion hypotheses; the length is the "
            f"log-likelihood ratio and the numbers are the share of positive and of negative rows that realise the hypothesis.")


@figure("pharm_hypotheses", GROUP, "The strongest hypotheses of the ligand-based pharmacophore model of each enzyme", hypotheses_caption, data=("data/tdi_track/outputs/caches/pharmacophore_hypotheses.npz",), section="9.14")
def hypotheses(theme_name):
    from cypmode.challenge import pharmacophore_hypotheses as ph

    th = apply(theme_name)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), constrained_layout=True)
    for ax, isoform in zip(axes, ("CYP3A4", "CYP2D6"), strict=True):
        info = ph.summary(isoform)
        rows = info["hypotheses"]
        for i, row in enumerate(rows):
            colour = th["green"] if row["side"] == "positive" else th["red"]
            ax.barh(i, row["ratio"], color=colour)
            ax.text(row["ratio"] + (0.03 if row["ratio"] > 0 else -0.03), i, f"{row['a'] / info['n_positive']:.0%} vs {row['b'] / info['n_negative']:.0%}", va="center", ha="left" if row["ratio"] > 0 else "right", fontsize=7, color=th["muted"])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["description"] or "" for r in rows], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color=th["muted"], linewidth=0.8)
        limit = max(abs(r["ratio"]) for r in rows) * 1.45
        ax.set_xlim(-limit, limit)
        ax.set_xlabel("log-likelihood ratio (positive vs negative rows)", fontsize=9)
        ax.set_title(f"{isoform}: {info['n_positive']} positive and {info['n_negative']} negative rows, {info['vocabulary']:,} hypotheses in the vocabulary", fontsize=10)
        tidy(ax, th, grid=False)
    return fig


def stacked_rows() -> list[dict]:
    """Rows for the panel of AUC changes: one per representation and enzyme, from the stacked comparison of its log (two cuts)."""
    sources = (
        ("physics block, 4 levels", "physics_experiment_{iso}.log", ("CYP3A4", "CYP2D6")),
        ("physics block, 20 columns", "physics_compact_{iso}.log", ("CYP3A4", "CYP2D6")),
        ("physics block, tier B", "physics_tierb_{iso}.log", ("CYP3A4", "CYP2D6")),
        ("pocket block, own pocket", "aux_pocket_{pocket}_{iso}.log", ("CYP3A4", "CYP2D6")),
        ("pocket block, other pocket (control)", "exploratory_pocket_control_{iso}.log", ("CYP3A4", "CYP2D6")),
        ("site model, own pocket", "aux_site_{pocket}_{iso}.log", ("CYP3A4", "CYP2D6")),
        ("site model, other pocket (control)", "exploratory_site_control_{iso}.log", ("CYP3A4", "CYP2D6")),
        ("hypothesis model", "aux_hypothesis_{iso}.log", ("CYP3A4", "CYP2D6")),
        ("hypothesis model, permuted labels (control)", "exploratory_hypothesis_permuted_{iso}.log", ("CYP3A4", "CYP2D6")),
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
    passing = [r for r in stacked if "(control)" not in r["label"] and all(p < 0.05 and g > 0 for g, p in r["cuts"].values()) and len(r["cuts"]) == 2]
    return (f"Every pharmacophore representation and model of the project and what happened to it (the fourth level of the physics block is its pharmacophore level; the site model and the hypothesis model are the two pharmacophore models, each with its control). Left: the change in AUC when the block is added to the default stack, at the two confidence cuts, for the "
            f"{len(stacked)} comparisons that have a log (a point is filled when its corrected p is below 0.05); the gate asks for both cuts, and {len(passing)} of these comparisons pass it. Middle: the change in MCC "
            f"of the older blocks against the same base, with the 95% interval and the Holm-adjusted p ({len(older)} comparisons). Right: the Spearman correlation of the pose terms with minus the true RMSD "
            f"over the benchmark poses, with the gate at {GATE_RHO} ({len(poses)} terms).")


@figure("pharm_forest", GROUP, "Every pharmacophore representation of the project and its verdict", forest_caption,
        data=("data/structure_track/outputs/cyp3a4/logs/physics_experiment_CYP3A4.log", "data/tdi_track/outputs/comparisons/pharmacophore_fingerprint_significance_paired.csv",
              "data/structure_track/outputs/cyp3a4/logs/pose_awareness.log"), section="9.14")
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
