# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Figures of the structure track from its real tables: RMSD and LDDT-PLI drawn on a real pose against its crystal pose, the pose pools of the crystal-anchored benchmark ligands, the LDDT-PLI of the
project against that of OpenStructure, the effect of the force-field relaxation on the margin of the poses, the triangle update of the pair representation, and (held back) the pose pools of the 20 official ligands."""

from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import figure
from scipy.stats import spearmanr
from style import apply, tidy

GROUP = "structure"
ROOT = Path(__file__).resolve().parents[3]
OUT = "data/structure_track/outputs/cyp3a4"
BENCHMARK = "data/structure_track/inputs/structure_benchmark.csv"
BOLTZ_POSES = f"{OUT}/benchmark_engine_pose_validation_unguided_boltz.csv"
SUCCESS_A = 2.0


def short(compound: str) -> str:
    return compound.removeprefix("bench-")


# -- RMSD and LDDT-PLI on a real pose --------------------------------------------------------------------------------------------------------------

def kabsch(moving: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation that place ``moving`` on ``fixed`` with the least squared distance."""
    mc, fc = moving.mean(axis=0), fixed.mean(axis=0)
    u, _s, vt = np.linalg.svd((moving - mc).T @ (fixed - fc))
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1, 1, d]) @ vt
    return rotation, fc - mc @ rotation


def lddt_pairs(model_ligand, model_protein, reference_ligand, reference_protein, permutations):
    """The distances of the ligand-protein pairs that LDDT-PLI looks at (protein atoms within the inclusion radius of the crystal ligand), for the symmetry mapping with the best score.
    Returns (score, reference distances, model distances)."""
    from cypmode.validation.pose_selection import INCLUSION_RADIUS_A, THRESHOLDS_A

    keys = sorted(set(model_protein) & set(reference_protein))
    model_xyz, reference_xyz = np.array([model_protein[k] for k in keys]), np.array([reference_protein[k] for k in keys])
    reference_d = np.linalg.norm(reference_ligand[:, None, :] - reference_xyz[None, :, :], axis=2)
    included = reference_d < INCLUSION_RADIUS_A
    columns = included.any(axis=0)
    model_d = np.linalg.norm(model_ligand[:, None, :] - model_xyz[None, columns, :], axis=2)
    reference_d, included = reference_d[:, columns], included[:, columns]
    best, best_permutation = -1.0, permutations[0]
    for permutation in permutations:
        error = np.abs(model_d[permutation] - reference_d)[included]
        score = float(np.mean([np.mean(error < t) for t in THRESHOLDS_A]))
        if score > best:
            best, best_permutation = score, permutation
    return best, reference_d[included], model_d[best_permutation][included]


@cache
def pose_study() -> dict:
    """The benchmark ligand whose best Boltz sample is a correct placement (within 2 A) and whose worst sample is furthest from the crystal pose; its best and its worst pose against the crystal ligand.
    A pose is compared with the crystal copy that gives it the highest LDDT-PLI, which is how the project scores it."""
    from rdkit import Chem

    from cypmode.validation.lddt_to_crystal import crystal_references
    from cypmode.validation.pose_pools import (
        pool_paths,
        template_automorphisms,
        template_ligand_coordinates,
    )
    from cypmode.validation.pose_selection import complex_arrays
    from cypmode.validation.structures import load_structure

    benchmark = pd.read_csv(ROOT / BENCHMARK)
    smiles_of = dict(zip(benchmark["Molecule_Name"], benchmark["SMILES"], strict=True))
    table = pd.read_csv(ROOT / BOLTZ_POSES)
    extremes = table.groupby("compound")["true_rmsd"].agg(["min", "max"])
    ligand = (extremes[extremes["min"] <= SUCCESS_A]["max"]).idxmax()
    smiles = smiles_of[ligand]
    references = crystal_references(smiles)
    rows = table[table["compound"] == ligand].set_index("sample")["true_rmsd"]
    paths = {sample: path for _engine, sample, path in pool_paths(ligand, ("boltz",))}
    permutations = template_automorphisms(smiles)
    poses = {}
    for role, sample in (("best", int(rows.idxmin())), ("worst", int(rows.idxmax()))):
        structure = load_structure(paths[sample])
        model_ligand, model_protein = template_ligand_coordinates(structure, smiles), complex_arrays(structure)[1]
        choice = None
        for reference in references:
            score, ref_d, model_d = lddt_pairs(model_ligand, model_protein, reference.ligand, reference.protein, permutations)
            if choice is None or score > choice["lddt"]:
                choice = {"reference": reference, "lddt": score, "reference_d": ref_d, "model_d": model_d}
        reference = choice["reference"]
        reference_ca = {k: v for k, v in reference.protein.items() if k[1] == "CA"}
        near = [k for k, v in reference_ca.items() if np.linalg.norm(reference.ligand - v, axis=1).min() < 10.0 and k in model_protein]
        rotation, shift = kabsch(np.array([model_protein[k] for k in near]), np.array([reference_ca[k] for k in near]))
        placed = model_ligand @ rotation + shift
        rmsds = [float(np.sqrt(np.mean(np.sum((placed[p] - reference.ligand) ** 2, axis=1)))) for p in permutations]
        poses[role] = {"sample": sample, "placed": placed, "permutation": permutations[int(np.argmin(rmsds))], "rmsd": min(rmsds), "table_rmsd": float(rows.loc[sample]), "lddt": choice["lddt"],
                       "reference_d": choice["reference_d"], "model_d": choice["model_d"], "reference": reference.ligand, "pdb": reference.pdb_id}
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in Chem.MolFromSmiles(smiles).GetBonds()]
    return {"ligand": ligand, "poses": poses, "bonds": bonds, "samples": len(rows)}


def pose_caption() -> str:
    study = pose_study()
    best, worst = study["poses"]["best"], study["poses"]["worst"]
    return (f"RMSD and LDDT-PLI on a real prediction. The ligand is {short(study['ligand'])}, the benchmark ligand whose best Boltz-2 sample is a correct placement and whose worst sample is furthest from the crystal pose (crystal entries {best['pdb']} and {worst['pdb']}). "
            f"Top: the heavy atoms of the crystal ligand (open circles) and of the predicted ligand (filled), both seen along the two principal axes of the crystal ligand after the pocket of the prediction has been superposed on the crystal pocket "
            f"(the C-alpha atoms within 10 A of the ligand, the superposition of BiSyRMSD); each dotted line joins an atom to its crystal counterpart, and the RMSD is the root of the mean squared length of these lines, taken over the symmetry-equivalent mappings of the ligand: "
            f"{best['rmsd']:.2f} A for sample {best['sample']} and {worst['rmsd']:.2f} A for sample {worst['sample']}. "
            f"Bottom: LDDT-PLI needs no superposition. Every protein atom within 6 A of the crystal ligand is paired with every ligand atom; a point is one pair, with its distance in the crystal on the horizontal axis and in the prediction on the vertical axis; "
            f"the bands are the four tolerances 0.5, 1, 2 and 4 A, and the score is the mean over the tolerances of the share of pairs inside the band: {best['lddt']:.2f} for the best sample and {worst['lddt']:.2f} for the worst. "
            "A pose that fails the PoseBusters checks scores zero whatever its geometry.")


@figure("struct_rmsd_lddt", GROUP, "RMSD and LDDT-PLI drawn on a real pose", pose_caption, data=(BOLTZ_POSES, f"{OUT}/pose_pool_cache", "data/structure_track/inputs/pdb_cache"), section="11.4")
def rmsd_lddt(theme_name):
    th = apply(theme_name)
    study = pose_study()
    top = float(np.ceil(max(pose["model_d"].max() for pose in study["poses"].values()) + 0.5))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0))
    for column, role in enumerate(("best", "worst")):
        pose = study["poses"][role]
        reference = pose["reference"]
        centre = reference.mean(axis=0)
        plane = np.linalg.svd(reference - centre)[2][:2].T
        color = th["blue"] if role == "best" else th["red"]
        ax = axes[0, column]
        flat_reference, flat_model = (reference - centre) @ plane, (pose["placed"] - centre) @ plane
        for a, b in study["bonds"]:
            ax.plot(flat_reference[[a, b], 0], flat_reference[[a, b], 1], color=th["muted"], lw=1.4, zorder=1)
            ax.plot(flat_model[[a, b], 0], flat_model[[a, b], 1], color=color, lw=1.8, zorder=2)
        matched = flat_model[pose["permutation"]]
        for i in range(len(flat_reference)):
            ax.plot([flat_reference[i, 0], matched[i, 0]], [flat_reference[i, 1], matched[i, 1]], color=th["gold"], lw=0.9, ls=":", zorder=1)
        ax.scatter(flat_reference[:, 0], flat_reference[:, 1], s=60, facecolor="none", edgecolor=th["text"], zorder=3, label="crystal")
        ax.scatter(flat_model[:, 0], flat_model[:, 1], s=46, color=color, zorder=3, label="prediction")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"sample {pose['sample']}: RMSD {pose['rmsd']:.2f} A", fontsize=11)
        ax.legend(frameon=False, loc="lower left", fontsize=8)
        ax = axes[1, column]
        x = np.linspace(0, 6.2, 50)
        for tolerance, alpha in ((4.0, 0.08), (2.0, 0.1), (1.0, 0.14), (0.5, 0.2)):
            ax.fill_between(x, x - tolerance, x + tolerance, color=th["green"], alpha=alpha, linewidth=0)
        ax.plot(x, x, color=th["muted"], lw=1)
        ax.scatter(pose["reference_d"], pose["model_d"], s=7, color=color, alpha=0.6, linewidths=0)
        ax.set(xlim=(0, 6.2), ylim=(0, top), xlabel="ligand-protein distance in the crystal (Å)", ylabel="in the prediction (Å)", title=f"LDDT-PLI {pose['lddt']:.2f}")
        tidy(ax, th)
    fig.tight_layout()
    return fig


# -- the pose pools of the benchmark ligands -------------------------------------------------------------------------------------------------------

def pool_caption() -> str:
    table, outcomes = pd.read_csv(ROOT / BOLTZ_POSES), pd.read_csv(ROOT / OUT / "selection_validation_outcomes.csv")
    n = outcomes["compound"].nunique()
    first = int((outcomes["first_boltz"] <= SUCCESS_A).sum())
    best = int((outcomes["oracle_boltz"] <= SUCCESS_A).sum())
    mbr = int((outcomes["mbr_boltz"] <= SUCCESS_A).sum())
    return (f"The pool of poses that Boltz-2 gives for each of the {n} crystal-anchored benchmark ligands ({int(table.groupby('compound').size().iloc[0])} samples each, unguided): one dot per sample, at its RMSD to the crystal pose (vertical axis), ligands ordered by the RMSD of their first sample. "
            f"The first sample, the model's own top choice, is the filled circle; the best sample in the pool is ringed in green, which no rule can see without the crystal; the diamond is the pick of minimum Bayes risk. The dotted line is 2 A, the usual limit of a correct placement. "
            f"The first sample is within 2 A for {first} of {n} ligands, the best sample of the pool for {best}, and the minimum-Bayes-risk pick for {mbr}: the pool often holds a good pose that nothing available at prediction time can single out, "
            "which is the whole difficulty of the structure track and the reason every selection rule was put to a gate.")


@figure("struct_pose_pools", GROUP, "The pose pools of the benchmark ligands", pool_caption, data=(BOLTZ_POSES, f"{OUT}/selection_validation_outcomes.csv"), section="11.8")
def pose_pools(theme_name):
    th = apply(theme_name)
    table, outcomes = pd.read_csv(ROOT / BOLTZ_POSES), pd.read_csv(ROOT / OUT / "selection_validation_outcomes.csv").set_index("compound")
    order = list(outcomes["first_boltz"].sort_values().index)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for position, compound in enumerate(order):
        part = table[table["compound"] == compound]
        ax.scatter(np.full(len(part), position), part["true_rmsd"], s=26, color=th["muted"], alpha=0.8, zorder=2)
        ax.scatter(position, outcomes.loc[compound, "oracle_boltz"], s=130, facecolor="none", edgecolor=th["green"], linewidth=2, zorder=3)
        ax.scatter(position, outcomes.loc[compound, "first_boltz"], s=62, color=th["blue"], zorder=4)
        ax.scatter(position, outcomes.loc[compound, "mbr_boltz"], s=70, marker="D", color=th["orange"], zorder=3)
    ax.axhline(SUCCESS_A, color=th["muted"], ls=":", lw=1.2)
    ax.set_xticks(range(len(order)), [short(c) for c in order], rotation=40, fontsize=8)
    ax.set(ylabel="RMSD to the crystal pose (Å)", title="one dot per sample of the pool")
    ax.scatter([], [], s=62, color=th["blue"], label="first sample (the model's choice)")
    ax.scatter([], [], s=130, facecolor="none", edgecolor=th["green"], linewidth=2, label="best sample of the pool")
    ax.scatter([], [], s=70, marker="D", color=th["orange"], label="minimum Bayes risk pick")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- the LDDT-PLI of the project against OpenStructure ---------------------------------------------------------------------------------------------

def ost_caption() -> str:
    from cypmode.validation.ost_official import (
        MAX_MEAN_ABSOLUTE_DIFFERENCE,
        MIN_PAIR_AGREEMENT,
        MIN_RANK_CORRELATION,
        pairs_agreement,
    )

    table = pd.read_csv(ROOT / OUT / "ost_vs_ours.csv").dropna(subset=["ours", "ost"])
    rho = spearmanr(table["ours"], table["ost"])[0]
    mad = float((table["ours"] - table["ost"]).abs().mean())
    agreement = pairs_agreement(table)
    return (f"The LDDT-PLI of the project against the LDDT-PLI of OpenStructure, the scorer of the organisers, for {len(table)} predicted poses of crystal-anchored ligands (one point per pose; the line is equality). "
            f"The two rank the poses almost the same way overall (Spearman {rho:.3f}; registered bar {MIN_RANK_CORRELATION}) but the local value differs from the official one by {mad:.3f} on average (bar {MAX_MEAN_ABSOLUTE_DIFFERENCE}), "
            f"and of the pairs of poses of one ligand {agreement:.1%} are ordered the same way by both (bar {MIN_PAIR_AGREEMENT:.0%}); the agreement gate is therefore not met, and the structure gates were repeated with the official yardstick (section 11.8). "
            "The local measure follows the definition of the official one but differs from it in details of the protein atoms and ligand atoms that enter the score, which is the likely source of the offset.")


@figure("struct_ost_vs_project", GROUP, "The LDDT-PLI of the project against OpenStructure", ost_caption, data=(f"{OUT}/ost_vs_ours.csv",), section="11.4")
def ost_vs_project(theme_name):
    th = apply(theme_name)
    table = pd.read_csv(ROOT / OUT / "ost_vs_ours.csv").dropna(subset=["ours", "ost"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=th["muted"], lw=1)
    ax.scatter(table["ost"], table["ours"], s=16, color=th["blue"], alpha=0.7, linewidths=0)
    ax.set(xlabel="LDDT-PLI by OpenStructure", ylabel="LDDT-PLI of the project", title=f"{len(table)} poses, Spearman {spearmanr(table['ours'], table['ost'])[0]:.3f}", xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("equal")
    tidy(ax, th)
    ax = axes[1]
    difference = (table["ours"] - table["ost"]).to_numpy()
    ax.hist(difference, bins=24, color=th["orange"])
    ax.axvline(0, color=th["text"], lw=1)
    ax.axvline(difference.mean(), color=th["red"], ls="--", lw=1.5, label=f"mean {difference.mean():+.3f}")
    ax.set(xlabel="project minus OpenStructure", ylabel="poses", title="the offset of the local measure")
    ax.legend(frameon=False)
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- the force-field relaxation and the margin of the poses ----------------------------------------------------------------------------------------

MARGIN_FLOOR = 0.8


def margin_caption() -> str:
    table = pd.read_csv(ROOT / OUT / "relaxation_gate_poses.csv")
    both = table.dropna(subset=["margin_before", "margin_after"])
    pairs = both.dropna(subset=["lddt_pli_before", "lddt_pli_after"])
    return (f"The force-field relaxation of the ligand in the frozen pocket, on {len(table)} Boltz-2 poses of {table['compound'].nunique()} crystal-anchored ligands. Left: the smallest relative distance between the ligand and the receptor (the margin of the scorer's PoseBusters call: "
            f"the check fails below 0.75 and the project calls a pose safe from {MARGIN_FLOOR}), before and after the relaxation, for the {len(both)} poses that pass the checks both times; the share of poses passing all 22 checks rises from "
            f"{table['passed_before'].mean():.1%} to {table['passed_after'].mean():.1%}, and the median margin from {both['margin_before'].median():.3f} to {both['margin_after'].median():.3f}. "
            f"Right: the LDDT-PLI against the crystal pose before and after, for the {len(pairs)} poses that have both: the relaxation moves the ligand by {table['rmsd'].median():.2f} A at the median and leaves the agreement with the crystal pose unchanged on average "
            f"(change {(pairs['lddt_pli_after'] - pairs['lddt_pli_before']).mean():+.4f}).")


@figure("struct_relaxation_margin", GROUP, "Pose margins and agreement before and after the relaxation", margin_caption, data=(f"{OUT}/relaxation_gate_poses.csv",), section="11.10")
def relaxation_margin(theme_name):
    th = apply(theme_name)
    table = pd.read_csv(ROOT / OUT / "relaxation_gate_poses.csv")
    both = table.dropna(subset=["margin_before", "margin_after"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax = axes[0]
    ax.plot([0.6, 1.0], [0.6, 1.0], color=th["muted"], lw=1)
    ax.axhline(MARGIN_FLOOR, color=th["gold"], ls=":", lw=1.3)
    ax.axvline(MARGIN_FLOOR, color=th["gold"], ls=":", lw=1.3)
    ax.axhline(0.75, color=th["red"], ls=":", lw=1.3)
    ax.axvline(0.75, color=th["red"], ls=":", lw=1.3)
    ax.scatter(both["margin_before"], both["margin_after"], s=18, color=th["blue"], alpha=0.75, linewidths=0)
    ax.set(xlabel="margin before the relaxation", ylabel="margin after", title="the margin of the pose", xlim=(0.6, 1.0), ylim=(0.6, 1.0))
    ax.text(0.755, 0.605, "check fails below", color=th["red"], fontsize=8, ha="right")
    ax.text(0.805, 0.605, "safe from", color=th["gold"], fontsize=8)
    tidy(ax, th)
    ax = axes[1]
    pairs = both.dropna(subset=["lddt_pli_before", "lddt_pli_after"])
    ax.plot([0, 1], [0, 1], color=th["muted"], lw=1)
    ax.scatter(pairs["lddt_pli_before"], pairs["lddt_pli_after"], s=18, color=th["green"], alpha=0.75, linewidths=0)
    ax.set(xlabel="LDDT-PLI before", ylabel="after", title="agreement with the crystal pose", xlim=(0, 1), ylim=(0, 1))
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- the triangle update of the pair representation ------------------------------------------------------------------------------------------------

POSITIONS, CHANNELS = 7, 8


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


@cache
def triangle_state(smiles: str = "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F"):
    """The pair state of the first seven heavy atoms of fluconazole: the real distance of every pair expanded on eight radial functions, and one outgoing triangle update with fixed random weights."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    xyz = mol.GetConformer().GetPositions()[:POSITIONS]
    d = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)
    centres = np.linspace(1.0, 6.0, CHANNELS)
    z = np.exp(-((d[..., None] - centres) ** 2) / 1.5)
    rng = np.random.default_rng(0)
    w = {k: rng.standard_normal((CHANNELS, CHANNELS)) * 0.4 for k in ("a", "b", "ag", "bg", "g", "o")}

    def update(state):
        a, b = (state @ w["a"]) * sigmoid(state @ w["ag"]), (state @ w["b"]) * sigmoid(state @ w["bg"])
        mixed = np.einsum("ikc,jkc->ijc", a, b)
        normed = (mixed - mixed.mean(-1, keepdims=True)) / (mixed.std(-1, keepdims=True) + 1e-6)
        return sigmoid(state @ w["g"]) * (normed @ w["o"])

    update_z = update(z)
    relabel = np.random.default_rng(1).permutation(POSITIONS)
    equivariance = float(np.abs(update(z[np.ix_(relabel, relabel)]) - update_z[np.ix_(relabel, relabel)]).max())
    return d, z, update_z, equivariance


def triangle_caption() -> str:
    d, _z, _update_z, equivariance = triangle_state()
    return (f"The triangle update of the Pairformer, the step of a co-folding model that makes its picture of the distances between positions consistent. Left: three atoms of fluconazole and their real distances, d(i,j) = {d[0, 1]:.2f} A, d(i,k) = {d[0, 2]:.2f} A and d(k,j) = {d[2, 1]:.2f} A; "
            f"the triangle inequality d(i,j) <= d(i,k) + d(k,j) holds ({d[0, 1]:.2f} <= {d[0, 2] + d[2, 1]:.2f}), so what is known about the pairs (i,k) and (k,j) limits the pair (i,j). Middle: the pair state of the first {POSITIONS} atoms, the real distance of each pair spread over {CHANNELS} channels "
            f"(shown as the sum of the channels). Right: the same state after one outgoing triangle update, in which the entry (i,j) is rebuilt from the products of the entries (i,k) and (j,k) over every third position k, with gates and a normalisation; the weights are random with a fixed seed, because the figure shows the computation and not a trained model. "
            f"Relabelling the positions changes the result by {equivariance:.0e}, which is rounding: the update has no preferred order of positions.")


@figure("struct_pairformer_triangle", GROUP, "The triangle update of the pair representation", triangle_caption, data=("the structure of fluconazole (the ligand of 6MA7)",), section="11.2")
def pairformer_triangle(theme_name):
    th = apply(theme_name)
    d, z, update_z, _ = triangle_state()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
    ax = axes[0]
    ax.axis("off")
    points = {"i": (0.1, 0.25), "j": (0.9, 0.25), "k": (0.5, 0.85)}
    for a, b, length in (("i", "j", d[0, 1]), ("i", "k", d[0, 2]), ("k", "j", d[2, 1])):
        ax.plot([points[a][0], points[b][0]], [points[a][1], points[b][1]], color=th["red"] if (a, b) == ("i", "j") else th["blue"], lw=2.4)
        offset = (-0.11, 0.0) if (a, b) == ("i", "k") else (0.11, 0.0) if (a, b) == ("k", "j") else (0.0, -0.09)
        ax.text((points[a][0] + points[b][0]) / 2 + offset[0], (points[a][1] + points[b][1]) / 2 + offset[1], f"{length:.2f} A", color=th["text"], fontsize=10, ha="center")
    for name, (x, y) in points.items():
        ax.scatter(x, y, s=700, color=th["panel"], edgecolor=th["text"], zorder=3)
        ax.text(x, y, name, ha="center", va="center", fontsize=13, color=th["text"], zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.05)
    ax.set_title("the triangle inequality")
    for ax, state, title in ((axes[1], z.sum(-1), "pair state: the real distances"), (axes[2], update_z.sum(-1), "after one triangle update")):
        image = ax.imshow(state, cmap="viridis")
        ax.set_xticks(range(POSITIONS), [str(i) for i in range(POSITIONS)])
        ax.set_yticks(range(POSITIONS), [str(i) for i in range(POSITIONS)])
        ax.set(xlabel="position j", ylabel="position i", title=title)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


# -- the pose pools of the official ligands (held back) -------------------------------------------------------------------------------------------

OFFICIAL_POOLS = f"{OUT}/final/official_posebusters_pools.csv"
RELAXATION = f"{OUT}/final/relaxation/decisions.csv"


def official_caption() -> str:
    pools = pd.read_csv(ROOT / OFFICIAL_POOLS)
    decisions = pd.read_csv(ROOT / RELAXATION)
    valid = pools[pools["passed"].astype(bool)]
    return (f"The pools of poses of the {pools['ligand'].nunique()} official ligands under the scorer's own PoseBusters call: {len(pools)} poses in all, {len(valid)} of them passing the 22 checks and {int((valid['distance_margin'] >= MARGIN_FLOOR).sum())} passing with a margin of at least {MARGIN_FLOOR}. "
            f"Left: for every ligand, how many poses of its pool fail a check (red), pass with a small margin (gold) and pass with a safe margin (green). Right: the margin of the pose that the recipe chose, before the force-field relaxation (open circle) and after it (filled), "
            f"for the same ligands in the same order; the dotted lines are the limit of the check (0.75) and the floor of the project ({MARGIN_FLOOR}); the relaxation raises the margin for {int((decisions['margin_after'] > decisions['margin_before']).sum())} of {len(decisions)} ligands. "
            "This figure shows predictions for the blinded compounds and is held back until the challenge closes.")


@figure("struct_official20_pools", GROUP, "The pose pools of the official ligands (held back)", official_caption, data=(OFFICIAL_POOLS, RELAXATION), section="11.10")
def official_pools(theme_name):
    from matplotlib.patches import Patch

    th = apply(theme_name)
    pools, decisions = pd.read_csv(ROOT / OFFICIAL_POOLS), pd.read_csv(ROOT / RELAXATION).set_index("compound")
    order = list(pools.groupby("ligand")["distance_margin"].max().sort_values().index)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), sharey=True, gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    for row, ligand in enumerate(order):
        part = pools[pools["ligand"] == ligand]
        passed = part["passed"].astype(bool)
        failed, small = int((~passed).sum()), int((passed & (part["distance_margin"] < MARGIN_FLOOR)).sum())
        safe = int((passed & (part["distance_margin"] >= MARGIN_FLOOR)).sum())
        ax.barh(row, failed, color=th["red"])
        ax.barh(row, small, left=failed, color=th["gold"])
        ax.barh(row, safe, left=failed + small, color=th["green"])
    ax.set_yticks(range(len(order)), order, fontsize=7.5)
    ax.set(xlabel="poses in the pool", title="what each pool holds")
    ax.legend(handles=[Patch(color=th["red"], label="fails a check"), Patch(color=th["gold"], label=f"passes, margin below {MARGIN_FLOOR}"), Patch(color=th["green"], label=f"passes, margin {MARGIN_FLOOR} or more")], frameon=False, fontsize=8, loc="lower right")
    tidy(ax, th, grid=False)
    ax = axes[1]
    for row, ligand in enumerate(order):
        if ligand in decisions.index:
            before, after = decisions.loc[ligand, "margin_before"], decisions.loc[ligand, "margin_after"]
            ax.plot([before, after], [row, row], color=th["muted"], lw=1.4, zorder=1)
            ax.scatter(before, row, s=40, facecolor="none", edgecolor=th["blue"], zorder=2)
            ax.scatter(after, row, s=40, color=th["blue"], zorder=3)
    ax.axvline(MARGIN_FLOOR, color=th["gold"], ls=":", lw=1.4)
    ax.axvline(0.75, color=th["red"], ls=":", lw=1.4)
    ax.set(xlabel="smallest relative distance to the receptor", title="the chosen pose: before (open) and after (filled) the relaxation")
    tidy(ax, th)
    fig.tight_layout()
    return fig
