# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The pharmacophore atlas of the TDI track (checklist item "Pharmacophore atlas", declared on 2026-10-23 before any code of it ran): every representation and model drawn on real molecules in 2D and in 3D,
with the functional groups and the structure-activity relationships.

No blinded compound is involved. The functional groups come from ``cypmode.challenge.functional_groups`` (a fixed table); the examples are chosen by the rules written in the checklist item, not by eye; every 2D
drawing is made by RDKit from a SMILES, every 3D picture from real coordinates (a crystal entry, or an ETKDG conformer with the seed 42 where no structure exists, said in the caption).
"""

import functools
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import figure
from matplotlib.patches import Patch
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from style import THEMES, apply, tidy

from cypmode.challenge import functional_groups as fg
from cypmode.challenge import pocket_pharmacophore as pp
from gen.molecules import draw, hex_to_unit

GROUP = "pharmacophore"
ROOT = Path(__file__).resolve().parents[3]
TRAIN = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
ISOFORMS = ("CYP3A4", "CYP2D6")
FAMILY_COLOUR = {"Donor": "blue", "Acceptor": "red", "Aromatic": "green", "Hydrophobe": "orange", "PosIonizable": "purple", "NegIonizable": "teal"}
FAMILY_LABEL = {"Donor": "donor", "Acceptor": "acceptor", "Aromatic": "aromatic ring", "Hydrophobe": "hydrophobe", "PosIonizable": "positive", "NegIonizable": "negative"}
GROUP_COLOURS = ("blue", "red", "green", "orange", "purple", "teal", "gold")
ELEMENT_COLOUR = {"C": "muted", "N": "blue", "O": "red", "S": "gold", "Cl": "green", "F": "teal", "Br": "orange"}


# -- shared helpers ------------------------------------------------------------------------------------------------------------------------------------------

def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name)


def training_pool(isoform: str) -> pd.DataFrame:
    """The labelled compounds of one enzyme (both arms measured): name, SMILES, label (1 = time-dependent inhibitor), the shift and the confidence of the label (|z|)."""
    from cypmode.challenge.label_confidence import confidence_z
    from cypmode.challenge.tdi_targets import BOTH_ARMS, training_view

    train = _read(TRAIN)
    view = training_view(train, isoform)
    keep = (view["population"] == BOTH_ARMS).to_numpy()
    return pd.DataFrame({"row": np.flatnonzero(keep), "name": train["Molecule_Name"][keep].to_numpy(), "smiles": train["SMILES"][keep].to_numpy(), "label": view["label"][keep].astype(int).to_numpy(),
                         "shift": view["shift"][keep].to_numpy(dtype=float), "z": confidence_z(train, isoform)[keep]})


def feature_atoms(mol) -> dict[str, list[tuple[int, ...]]]:
    """{family: [atoms of each feature]} of a molecule by the RDKit feature definitions (a topological view, no 3D coordinates needed)."""
    out: dict[str, list[tuple[int, ...]]] = {family: [] for family in pp.FAMILIES}
    for feature in pp.feature_factory().GetFeaturesForMol(mol):
        if feature.GetFamily() in out:
            out[feature.GetFamily()].append(tuple(feature.GetAtomIds()))
    return out


def _png(drawer) -> np.ndarray:
    drawer.FinishDrawing()
    return np.asarray(Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB"))


def draw_families(mol, theme_name: str, size=(520, 400)) -> tuple[np.ndarray, list[str]]:
    """A depiction in which every atom of a pharmacophore feature carries the colour of its family (an atom in several features carries several colours); the families that the molecule has are returned."""
    theme = THEMES[theme_name]
    found = feature_atoms(mol)
    atom_map: dict[int, list] = {}
    for family, groups in found.items():
        colour = hex_to_unit(theme[FAMILY_COLOUR[family]])
        for atoms in groups:
            for atom in atoms:
                if colour not in atom_map.setdefault(atom, []):
                    atom_map[atom].append(colour)
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    options = drawer.drawOptions()
    if theme_name == "dark":
        rdMolDraw2D.SetDarkMode(options)
    options.setBackgroundColour(hex_to_unit(theme["background"]))
    options.bondLineWidth = 2.0
    drawer.DrawMoleculeWithHighlights(mol, "", atom_map, {}, {a: 0.38 for a in atom_map}, {})
    return _png(drawer), [family for family in pp.FAMILIES if found[family]]


def draw_groups(mol, theme_name: str, size=(520, 400)) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """A depiction in which the atoms of each functional group carry one colour; the groups found and their colours (theme keys) are returned."""
    theme = THEMES[theme_name]
    found = fg.groups_of(mol)
    colours, bonds, legend = {}, {}, []
    for k, (name, matches) in enumerate(found.items()):
        key = GROUP_COLOURS[k % len(GROUP_COLOURS)]
        legend.append((name, key))
        for match in matches:
            for atom in match:
                colours[atom] = hex_to_unit(theme[key])
            for bond in mol.GetBonds():
                if bond.GetBeginAtomIdx() in match and bond.GetEndAtomIdx() in match:
                    bonds[bond.GetIdx()] = hex_to_unit(theme[key])
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    options = drawer.drawOptions()
    if theme_name == "dark":
        rdMolDraw2D.SetDarkMode(options)
    options.setBackgroundColour(hex_to_unit(theme["background"]))
    options.bondLineWidth = 2.0
    drawer.DrawMolecule(mol, highlightAtoms=list(colours), highlightBonds=list(bonds), highlightAtomColors=colours, highlightBondColors=bonds)
    return _png(drawer), legend


def family_legend(ax, th, families, fontsize=7):
    handles = [Patch(facecolor=th[FAMILY_COLOUR[f]], label=FAMILY_LABEL[f]) for f in families]
    ax.legend(handles=handles, loc="lower center", ncol=min(len(handles), 3), frameon=False, fontsize=fontsize, handlelength=1.0, columnspacing=0.8, bbox_to_anchor=(0.5, -0.1))


def group_legend(ax, th, legend, fontsize=7):
    handles = [Patch(facecolor=th[key], label=name) for name, key in legend]
    ax.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=fontsize, handlelength=1.0, columnspacing=0.8, bbox_to_anchor=(0.5, -0.12))


def blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


# -- the functional groups and the SAR of the label ----------------------------------------------------------------------------------------------------------

def group_table(isoform: str) -> pd.DataFrame:
    pool = training_pool(isoform)
    return fg.group_rates(fg.group_matrix(pool["smiles"].tolist()), pool["label"].to_numpy())


def groups_caption() -> str:
    tables = {iso: group_table(iso) for iso in ISOFORMS}
    top = {iso: tables[iso].iloc[0] for iso in ISOFORMS}
    acid = tables["CYP3A4"][tables["CYP3A4"].group == "carboxylic acid"]
    azole = tables["CYP3A4"][tables["CYP3A4"].group == "azole or pyridine (heme binder)"]
    return ("Which functional groups go with time-dependent inhibition? For each group of a fixed table (drawn at the left on a small example, with the pharmacophore features it carries), the share of time-dependent inhibitors "
            "among the training compounds that carry it, for CYP3A4 and CYP2D6, with the Wilson 95% interval; the dashed line is the share among all compounds and the number is how many carry the group (groups carried by at "
            f"least 10 compounds). The highest rate is for {top['CYP3A4'].group} on CYP3A4 ({top['CYP3A4'].rate:.0%} of {int(top['CYP3A4'].n)}) and for {top['CYP2D6'].group} on CYP2D6 ({top['CYP2D6'].rate:.0%} of {int(top['CYP2D6'].n)}), "
            "both groups that P450s are known to turn into reactive species. " + (f"Carboxylic acids are almost never inhibitors of CYP3A4 ({acid.iloc[0].rate:.0%} of {int(acid.iloc[0].n)}). " if len(acid) else "")
            + (f"The groups that bind the heme (azoles and pyridines) are not enriched ({azole.iloc[0].rate:.0%}): they inhibit by binding, which the time-dependent label is built to remove." if len(azole) else ""))


@figure("pharm_functional_groups", GROUP, "Which functional groups go with time-dependent inhibition", groups_caption, data=(TRAIN,), section="9.15")
def functional_groups(theme_name):
    th = apply(theme_name)
    tables = {iso: group_table(iso).set_index("group") for iso in ISOFORMS}
    order = [g for g in tables["CYP3A4"].sort_values("rate", ascending=False).index if g in tables["CYP2D6"].index] + [g for g in tables["CYP3A4"].index if g not in tables["CYP2D6"].index]
    rows = len(order)
    fig = plt.figure(figsize=(11.0, 0.58 * rows + 1.4), constrained_layout=True)
    grid = fig.add_gridspec(rows, 3, width_ratios=[0.9, 2.2, 2.2], hspace=0.05)
    for i, name in enumerate(order):
        thumb = fig.add_subplot(grid[i, 0])
        mol = Chem.MolFromSmiles(fg.EXAMPLES[name])
        match = fg.groups_of(mol).get(name, [])
        atoms = sorted({a for m in match for a in m})
        thumb.imshow(draw(mol, theme_name, atoms=atoms, size=(300, 110), line_width=1.6))
        blank(thumb)
        thumb.set_ylabel(textwrap.fill(name, 20), rotation=0, ha="right", va="center", fontsize=8)
        for k, iso in enumerate(ISOFORMS):
            ax = fig.add_subplot(grid[i, 1 + k])
            if name in tables[iso].index:
                row = tables[iso].loc[name]
                ax.barh(0, row.rate, color=th["blue"] if iso == "CYP3A4" else th["orange"], height=0.6)
                ax.errorbar(row.rate, 0, xerr=[[row.rate - row.low], [row.high - row.rate]], color=th["muted"], capsize=2, linewidth=1)
                ax.text(min(row.high + 0.02, 0.98), 0, f"{int(row.n)}", va="center", fontsize=7, color=th["muted"])
                ax.axvline(row.base_rate, color=th["muted"], linestyle="--", linewidth=0.8)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            tidy(ax, th, grid=False)
            if i < rows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(f"share of time-dependent inhibitors, {iso}", fontsize=8)
    return fig


# -- the examples with their groups and features -------------------------------------------------------------------------------------------------------------

def example_compounds(isoform: str) -> pd.DataFrame:
    """The rule of the checklist: the two positive and the two negative training compounds with the largest label confidence |z|, in that order (positives first)."""
    pool = training_pool(isoform).dropna(subset=["z"])
    picks = [pool[pool.label == label].sort_values("z", ascending=False).head(2) for label in (1, 0)]
    return pd.concat(picks, ignore_index=True)


def examples_caption() -> str:
    return ("The compounds that the label is surest about, with what they carry. For each enzyme the two time-dependent inhibitors and the two non-inhibitors of the training set with the largest label confidence, drawn from their "
            "SMILES. Upper row of each enzyme: the atoms of the functional groups of the fixed table (one colour per group); lower row: the atoms of the pharmacophore features by the RDKit definitions (a colour per family; "
            "an atom in two features, such as a ring nitrogen that is an acceptor and part of an aromatic ring, carries both). This is the 2D view of what the pocket block, the site model and the hypothesis model "
            "describe in 3D.")


@figure("pharm_features_examples", GROUP, "The surest compounds with their functional groups and their pharmacophore features", examples_caption, data=(TRAIN,), section="9.15")
def features_examples(theme_name):
    th = apply(theme_name)
    fig, axes = plt.subplots(4, 4, figsize=(13.0, 13.5), constrained_layout=True, gridspec_kw={"height_ratios": [1, 1, 1, 1]})
    for e, iso in enumerate(ISOFORMS):
        examples = example_compounds(iso)
        for k, row in examples.iterrows():
            mol = Chem.MolFromSmiles(row.smiles)
            groups_image, legend = draw_groups(mol, theme_name)
            families_image, families = draw_families(mol, theme_name)
            for r, (image, legend_function, payload) in enumerate(((groups_image, group_legend, legend), (families_image, family_legend, families))):
                ax = axes[2 * e + r][k]
                ax.imshow(image)
                blank(ax)
                legend_function(ax, th, payload)
            axes[2 * e][k].set_title(f"{iso}: {'inhibitor' if row.label else 'non-inhibitor'}\nshift {row['shift']:+.2f}, confidence z {row.z:.1f}", fontsize=8)
    return fig


# -- the SAR of a change: matched molecular pairs --------------------------------------------------------------------------------------------------------------

MIN_PAIRS, PAIRS_SHOWN = 5, 6


def pair_table(isoform: str):
    """The SAR of a change at the level of the functional group (``functional_groups.group_gain_pairs``) for the labelled compounds of one enzyme, with the pairs of the cache that ``mmp_transfer`` built for the
    feature store. At the level of the fragment exchange itself the training set has at most four labelled pairs per transformation, fewer than the minimum, which is why the pairs are grouped by the group they gain."""
    from cypmode.challenge.mmp_transfer import get_pairs

    smiles = list(_read(TRAIN)["SMILES"]) + list(_read(TEST)["SMILES"])  # the list of the feature store, so that the cache of the pairs is the one the model used
    pool = training_pool(isoform)
    return fg.group_gain_pairs(smiles, get_pairs(smiles), pool["row"].to_numpy(), pool["label"].to_numpy(), MIN_PAIRS)


def typical_pair(k: int, mean: float, pairs: dict) -> tuple[int, int, int]:
    """The pair that gains group ``k`` whose change of label is closest to the mean of the group (the most typical, not the most extreme; ties go to the lowest index): (row of the first compound, row of the
    second, id of its transformation)."""
    index = np.flatnonzero(pairs["gained"] == k)
    best = index[np.argmin(np.abs(pairs["delta"][index] - mean))]
    return int(pairs["a"][best]), int(pairs["b"][best]), int(pairs["t"][best])


def fragment_atoms(mol, fragment: str) -> list[int]:
    """The atoms of a molecule that make up a variable fragment of a matched pair (the attachment dummy excluded), found by cutting each acyclic single bond in turn and keeping the part whose canonical SMILES is
    the fragment; empty when no cut gives it."""
    target_mol = Chem.MolFromSmiles(fragment.replace("[*:1]", "*"))
    if target_mol is None:
        return []
    target = Chem.MolToSmiles(target_mol)
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        cut = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True, dummyLabels=[(0, 0)])
        try:
            parts = Chem.GetMolFrags(cut, asMols=True, sanitizeFrags=True)
        except Exception:
            continue
        for part, atoms in zip(parts, Chem.GetMolFrags(cut), strict=True):
            if Chem.MolToSmiles(part) == target:
                return [a for a in atoms if a < mol.GetNumAtoms()]
    return []


def feature_change(mol_a, mol_b) -> str:
    """The change in the number of pharmacophore features from the first compound to the second, as text."""
    a, b = feature_atoms(mol_a), feature_atoms(mol_b)
    parts = [f"{len(b[f]) - len(a[f]):+d} {FAMILY_LABEL[f]}" for f in pp.FAMILIES if len(b[f]) != len(a[f])]
    return ", ".join(parts) if parts else "no change in the features"


def mmp_caption() -> str:
    lines = []
    for iso in ISOFORMS:
        table, pairs = pair_table(iso)
        lines.append(f"{iso}: {int((pairs['gained'] >= 0).sum())} labelled pairs gain exactly one group; {len(table)} groups have at least {MIN_PAIRS} of them" + (f" ({int(table.head(PAIRS_SHOWN).pairs.min())} to {int(table.head(PAIRS_SHOWN).pairs.max())} in the six shown)" if len(table) else ""))
    return ("Structure-activity relationship as a change. A matched pair is two compounds that differ by one small fragment at the same place (single-bond fragmentation, a variable part of at most 10 heavy atoms). "
            "Each pair that gains one functional group of the fixed table and changes no other is counted for that group; the bars give the mean change of the label (0 = not a time-dependent inhibitor, 1 = inhibitor) over "
            "those pairs for the six groups with the most pairs, with one standard error. Below, for each group, one real pair, the one whose change is closest to the mean of its group, with the changed fragment highlighted "
            "and the change in the number of pharmacophore features. " + "; ".join(lines) + ". A change of label that follows a change in a feature is a structure-activity relationship; one that does not is not, "
            "and the pairs that are few say little: CYP2D6 has far fewer labelled pairs than CYP3A4.")


@figure("pharm_mmp_sar", GROUP, "Structure-activity relationship as a change: matched pairs of the training set", mmp_caption, data=(TRAIN, TEST, "data/tdi_track/outputs/caches/mmp_pairs.npz"), section="9.15")
def mmp_sar(theme_name):
    th = apply(theme_name)
    fig = plt.figure(figsize=(14.0, 15.5), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, hspace=0.04)
    for e, iso in enumerate(ISOFORMS):
        table, pairs = pair_table(iso)
        top = table.head(PAIRS_SHOWN)
        inner = outer[e].subgridspec(3, 6, height_ratios=[1.0, 1.25, 1.25])
        ax = fig.add_subplot(inner[0, :])
        y = np.arange(len(top))
        ax.barh(y, top["mean"], xerr=1.96 * top["se"].fillna(0), color=[th["green"] if m > 0 else th["red"] for m in top["mean"]], alpha=0.8, error_kw={"ecolor": th["muted"], "capsize": 3})
        ax.set_yticks(y)
        ax.set_yticklabels([f"gain of {r.group}   ({r.pairs} pairs)" for r in top.itertuples()], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color=th["muted"], linewidth=0.8)
        ax.set_xlabel(f"{iso}: mean change of the label (95% interval)", fontsize=9)
        tidy(ax, th)
        for k, row in enumerate(top.itertuples()):
            source, target, tid = typical_pair(row.k, row.mean, pairs)
            before, after = Chem.MolFromSmiles(pairs["smiles"][source]), Chem.MolFromSmiles(pairs["smiles"][target])
            left, right = pairs["names"][tid].split(">>")
            r, c = divmod(k, 3)
            for side, (mol, fragment, which) in enumerate(((before, left, source), (after, right, target))):
                cell = fig.add_subplot(inner[1 + r, 2 * c + side])
                cell.imshow(draw(mol, theme_name, atoms=fragment_atoms(mol, fragment), size=(380, 300), line_width=1.6))
                blank(cell)
                cell.set_title(f"label {int(pairs['y'][which])}" + (f"\ngains {textwrap.shorten(row.group, 24, placeholder='...')}" if side else ""), fontsize=8)
                if side == 1:
                    cell.set_xlabel(feature_change(before, after), fontsize=7)
    return fig


# -- the hypotheses of the ligand-based model on molecules, in 2D and in 3D -----------------------------------------------------------------------------------

NAME_TO_FAMILY = {"aromatic": "Aromatic", "acceptor": "Acceptor", "donor": "Donor", "hydrophobe": "Hydrophobe", "positive": "PosIonizable", "negative": "NegIonizable"}
HYPOTHESES_PER_SIDE = 3


def draw_conformer(ax, located: dict, th: dict, label_distances: bool = True) -> None:
    """A conformer in 3D with the features of a located hypothesis as translucent spheres in the colour of their family, joined by dashed lines that carry the distances in ångströms."""
    xyz = located["xyz"]
    for i, j in located["bonds"]:
        ax.plot(*zip(xyz[i], xyz[j], strict=True), color=th["muted"], linewidth=1.1, alpha=0.8)
    for element in sorted(set(located["elements"])):
        mask = np.array([e == element for e in located["elements"]])
        ax.scatter(*xyz[mask].T, s=14, color=th[ELEMENT_COLOUR.get(element, "muted")], depthshade=False)
    points = np.array([f["xyz"] for f in located["features"]])
    for feature, point in zip(located["features"], points, strict=True):
        ax.scatter(*point, s=260, color=th[FAMILY_COLOUR[NAME_TO_FAMILY[feature["name"]]]], alpha=0.55, edgecolor=th["edge"], linewidth=0.6, depthshade=False)
    pairs = [(i, j) for i in range(len(points)) for j in range(i + 1, len(points))]
    for (i, j), distance in zip(pairs, located["distances"], strict=True):
        ax.plot(*zip(points[i], points[j], strict=True), color=th["text"], linestyle="--", linewidth=1.0)
        if label_distances:
            middle = (points[i] + points[j]) / 2
            ax.text(*middle, f"{distance:.1f}", fontsize=7, color=th["text"], ha="center")
    centre, radius = xyz.mean(axis=0), np.ptp(xyz, axis=0).max() / 2 + 1.0
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.set_axis_off()


def hypothesis_drawings(smiles: str, signature: int, theme_name: str, size=(360, 300)):
    """(2D image with the atoms of the features of the hypothesis coloured by family, the located hypothesis for the 3D drawing, a label of the feature names); None when no conformer realises it."""
    from cypmode.challenge import pharmacophore_hypotheses as hypotheses

    located = hypotheses.locate(smiles, signature)
    if located is None:
        return None
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))  # the atom indices of the located features are those of the molecule written from the canonical SMILES
    mol = Chem.MolFromSmiles(canonical)
    th = THEMES[theme_name]
    colours = {atom: hex_to_unit(th[FAMILY_COLOUR[NAME_TO_FAMILY[f["name"]]]]) for f in located["features"] for atom in f["atoms"]}
    return draw(mol, theme_name, atoms=list(colours), atom_colors=colours, size=size, line_width=1.6), located


def hypotheses_caption() -> str:
    return (f"What the ligand-based pharmacophore model looks for, on real compounds. For each enzyme the {HYPOTHESES_PER_SIDE} strongest hypotheses that favour a positive label (top rows) and the {HYPOTHESES_PER_SIDE} strongest exclusion "
            "hypotheses (bottom rows), each shown on the training compound that realises it and that the label is surest about: at the left the atoms of its two or three features coloured by family (green aromatic ring, "
            "red acceptor, blue donor, orange hydrophobe), at the right the same features as spheres on a 3D conformer (ETKDG, seed 42: a low-energy shape, not a bound pose), joined by dashed lines that carry the distances in "
            "Å. The model is fitted on all training rows of the enzyme, so this is a description of what it holds and not a measure of how well it predicts; the numbers above each pair are the log-likelihood ratio and the share of "
            "positive and of negative training compounds that realise the hypothesis.")


@figure("pharm_hypotheses_2d3d", GROUP, "The hypotheses of the ligand-based model, on compounds, in 2D and in 3D", hypotheses_caption, data=("data/tdi_track/outputs/caches/pharmacophore_hypotheses.npz", TRAIN), section="9.15")
def hypotheses_2d3d(theme_name):
    from cypmode.challenge import pharmacophore_hypotheses as hypotheses

    th = apply(theme_name)
    fig = plt.figure(figsize=(15.5, 13.0), constrained_layout=True)
    grid = fig.add_gridspec(4, 2 * HYPOTHESES_PER_SIDE)
    for e, iso in enumerate(ISOFORMS):
        info = hypotheses.summary(iso, per_side=HYPOTHESES_PER_SIDE)
        for side_index, side in enumerate(("positive", "exclusion")):
            chosen = [h for h in info["hypotheses"] if h["side"] == side]
            for k, hyp in enumerate(chosen):
                row = 2 * e + side_index
                drawn = hypothesis_drawings(hyp["smiles"], hyp["signature"], theme_name)
                ax2 = fig.add_subplot(grid[row, 2 * k])
                ax3 = fig.add_subplot(grid[row, 2 * k + 1], projection="3d")
                blank(ax2)
                if drawn is None:
                    ax2.text(0.5, 0.5, "not realised\nby this conformer set", ha="center", va="center", fontsize=8, transform=ax2.transAxes)
                    ax3.set_axis_off()
                    continue
                image, located = drawn
                ax2.imshow(image)
                draw_conformer(ax3, located, th)
                ax2.set_title(textwrap.fill(f"{iso}, {side}: {hyp['description']}", 46) + f"\nratio {hyp['ratio']:+.2f}; {hyp['a'] / info['n_positive']:.0%} of positives, {hyp['b'] / info['n_negative']:.0%} of negatives", fontsize=7, loc="left")
    return fig


# -- the pockets in 3D with the sites of the model (real crystal structures, PyMOL) -------------------------------------------------------------------------

def reference_record(pocket: str) -> dict:
    """The crystal entry that shows a pocket: CYP3A4 the ketoconazole complex 2V0M, CYP2D6 the entry whose ligand, among those with at least 15 heavy atoms, comes closest to the iron (ties: the best resolution)."""
    records = json.loads((ROOT / pp.POCKETS[pocket]).read_text())["ligands"]
    if pocket == "cyp3a4":
        return next(r for r in records if r["pdb_id"] == "2V0M" and r["comp_id"] == "KLN" and r["n_heavy"] >= 30)
    return min((r for r in records if r["n_heavy"] >= 15), key=lambda r: (r["min_fe_distance"], r["resolution"] or 99.0, r["pdb_id"]))


def world_sites(pocket: str, record: dict) -> list[tuple[str, float, float, float, float]]:
    """The sites of the model of a pocket in the coordinates of one crystal entry: (family, x, y, z, weight). The sites live in the frame of the heme; the frame of the entry carries them back."""
    import gemmi

    fields = pocket_fields_of(pocket)
    iron, xyz = np.array(record["fe"], dtype=float), np.array(record["coords"], dtype=float)
    frame = pp.heme_frame(gemmi.read_structure(str(ROOT / pp.PDB_CACHE / f"{record['pdb_id']}.cif")), iron, xyz)
    out = []
    for family, rows in pp.sites_of(fields).items():
        for site in rows:
            x, y, z = iron + frame.T @ site[:3]
            out.append((family, float(x), float(y), float(z), float(site[3])))
    return out


def pocket_fields_of(pocket: str) -> dict[str, np.ndarray]:
    with np.load(ROOT / "data/tdi_track/outputs/caches" / f"pocket_fields_{pocket}.npz") as archive:
        return {name: archive[name] for name in archive.files}


def rgb_list(hex_colour: str) -> list[float]:
    return list(hex_to_unit(hex_colour))


def pharmacophore_scene(source: Path, comp: str | None, iron: list[float] | None, theme_name: str, key: str, spheres: list[tuple], solid: list[tuple[str, float, float, float]] = (), size=(1500, 1100),
                        crystal: tuple[Path, str, list[float]] | None = None, zoom: float = 8):
    """A PyMOL render of the heme pocket of a structure with a model drawn in it. ``source`` is a crystal entry (``comp`` the three-letter code of its ligand, ``iron`` the position of the iron whose chain is kept) or a
    predicted complex (``comp`` None: the ligand is residue LIG1 or LIG and the complex has one iron). ``spheres`` are the sites of a model as translucent spheres coloured by family: (family, x, y, z, radius in Å)
    or (family, x, y, z, radius, transparency); ``solid`` the features of a ligand as small solid spheres. ``crystal`` is the crystal entry of the same ligand (path, code, iron): it is aligned on the protein of
    ``source`` and its ligand drawn in white, so that the predicted pose is seen against the experimental one."""
    from gen.pymol_scene import orient, render

    th = THEMES[theme_name]
    colours = "\n".join(f'cmd.set_color("fam_{family}", {rgb_list(th[FAMILY_COLOUR[family]])!r})' for family in pp.FAMILIES)
    lines = []
    for i, sphere in enumerate(spheres):
        family, x, y, z, radius = sphere[:5]
        transparency = sphere[5] if len(sphere) > 5 else 0.55
        lines.append(f'cmd.pseudoatom("sites", pos=[{x:.3f}, {y:.3f}, {z:.3f}], name="S{i}", resi="{i}", vdw={radius:.2f}, color="fam_{family}")')
        lines.append(f'cmd.set("sphere_transparency", {transparency}, "sites and name S{i}")')
    sites = "\n".join(lines)
    features = "\n".join(f'cmd.pseudoatom("feat", pos=[{x:.3f}, {y:.3f}, {z:.3f}], name="F{i}", resi="{i}", vdw=0.45, color="fam_{family}")' for i, (family, x, y, z) in enumerate(solid))

    def keep_chain(obj: str, code: str, position: list[float], tag: str) -> str:
        low, high = [c - 1.5 for c in position], [c + 1.5 for c in position]
        return (f'cmd.select("fe{tag}", "{obj} and elem FE and (x>{low[0]} and x<{high[0]} and y>{low[1]} and y<{high[1]} and z>{low[2]} and z<{high[2]})")\n'
                f'_c{tag} = []\ncmd.iterate("fe{tag}", "_c{tag}.append(chain)", space={{"_c{tag}": _c{tag}}})\n'
                f'cmd.select("keep{tag}", "{obj} and (chain " + _c{tag}[0] + " or (resn {code} within 14 of fe{tag}))")\ncmd.remove("{obj} and not keep{tag}")\n')

    if comp:
        prepare = keep_chain("s", comp, iron, "0") + f'cmd.select("lig", "s and resn {comp} within 14 of fe0")'
    else:
        prepare = 'cmd.select("lig", "s and (resn LIG1 or resn LIG)")'
    crystal_block = ""
    if crystal is not None:
        crystal_path, crystal_comp, crystal_iron = crystal
        crystal_block = (f'cmd.load({str(crystal_path)!r}, "cryst")\ncmd.remove("cryst and solvent")\n' + keep_chain("cryst", crystal_comp, crystal_iron, "c")
                         + 'cmd.align("cryst and polymer and name CA", "s and polymer and name CA")\n'
                         + f'cmd.hide("everything", "cryst"); cmd.select("cl", "cryst and resn {crystal_comp} within 14 of fec"); cmd.show("sticks", "cl")\n'
                         + 'cmd.color("white" if PAL["bg"] != "0xffffff" else "0x444444", "cl and elem C"); cmd.set("stick_radius", 0.24, "cl")')
    show_features = "cmd.show('spheres', 'feat'); cmd.set('sphere_scale', 1.0, 'feat')" if solid else ""
    body = f'''
{colours}
cmd.load({str(source)!r}, "s")
cmd.remove("solvent")
{prepare}
cmd.hide("everything", "s")
cmd.select("hem", "s and resn HEM")
cmd.select("pocket", "byres (s and polymer within 4.0 of lig) or ((s and polymer and name SG and resn CYS) within 4 of s and elem FE)")
cmd.show("cartoon", "byres (s and polymer within 15 of lig)"); cmd.color(PAL["protein"], "s and polymer"); cmd.set("cartoon_transparency", PAL["cartoon_t"])
cmd.show("sticks", "pocket"); cmd.set("stick_radius", 0.11, "pocket")
cmd.color(PAL["pocket_c"], "pocket and elem C"); cmd.color(PAL["n"], "pocket and elem N"); cmd.color(PAL["o"], "pocket and elem O"); cmd.color(PAL["s"], "pocket and elem S")
cmd.show("sticks", "lig"); cmd.color(PAL["ligand"], "lig and elem C"); cmd.set("stick_radius", 0.2, "lig")
cmd.show("sticks", "hem"); cmd.color(PAL["heme"], "hem and elem C"); cmd.set("stick_radius", 0.17, "hem")
cmd.show("spheres", "s and elem FE"); cmd.set("sphere_scale", 0.5, "s and elem FE"); cmd.color(PAL["iron"], "s and elem FE")
{crystal_block}
{sites}
cmd.show("spheres", "sites"); cmd.set("sphere_scale", 1.0, "sites")
{features}
{show_features}
{orient(zoom=zoom, tilt=14, ref="s and ")}
cmd.clip("slab", 14)
cmd.set("depth_cue", 0)
'''
    return render(key, body, size, theme_name, {"fe": "s and elem FE", "ligand": "lig"}, files=(source, *([crystal[0]] if crystal else [])))


def pocket_images(theme_name: str) -> dict[str, dict]:
    """{pocket: {"record", "scene", "n_sites", "ligands"}} for the two enzymes."""
    from gen.pymol_scene import autocrop

    out = {}
    for pocket in ("cyp3a4", "cyp2d6"):
        record = reference_record(pocket)
        sites = world_sites(pocket, record)
        scene = pharmacophore_scene(ROOT / pp.PDB_CACHE / f"{record['pdb_id']}.cif", record["comp_id"], record["fe"], theme_name, f"pharm_sites_{pocket}_{record['pdb_id']}", [(f, x, y, z, pp.SITE_RADIUS) for f, x, y, z, _ in sites])
        image = plt.imread(scene.png)
        out[pocket] = {"record": record, "n_sites": len(sites), "image": autocrop(image), "ligands": len(json.loads((ROOT / pp.POCKETS[pocket]).read_text())["ligands"])}
    return out


def pocket3d_caption() -> str:
    parts = []
    for pocket in ("cyp3a4", "cyp2d6"):
        record = reference_record(pocket)
        parts.append(f"{pocket.upper()}: entry {record['pdb_id']} with the ligand {record['comp_id']} ({sum(len(v) for v in pp.sites_of(pocket_fields_of(pocket)).values())} sites)")
    return ("The pocket pharmacophore model on the protein it was read from. Real crystal structures rendered with PyMOL: the heme (orange) with its iron, the lining residues, the ligand of the entry (teal) and, as translucent spheres "
            "of the tolerance radius of 1.5 Å, the sites of the structure-based model of 9.14, coloured by feature family (blue donor, red acceptor, green aromatic ring, orange hydrophobe, purple positive). The sites are "
            "built from the ligands of all the crystal entries of the enzyme in the frame of the heme and carried back into the coordinates of the entry shown. " + "; ".join(parts) + ". The model is holo: it knows where ligands "
            "sat in these complexes, not what an apo or a cryptic pocket looks like (see 11.11).")


@figure("pharm_pocket_3d", GROUP, "The pocket pharmacophore model on the crystal structures of CYP3A4 and CYP2D6", pocket3d_caption, data=("data/tdi_track/outputs/caches/pocket_fields_cyp3a4.npz", "data/tdi_track/outputs/caches/pocket_fields_cyp2d6.npz"), section="9.15")
def pocket_3d(theme_name):
    th = apply(theme_name)
    images = pocket_images(theme_name)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), constrained_layout=True)
    for ax, pocket in zip(axes, ("cyp3a4", "cyp2d6"), strict=True):
        item = images[pocket]
        ax.imshow(item["image"])
        blank(ax)
        ax.set_title(f"{pocket.upper()}: entry {item['record']['pdb_id']}, ligand {item['record']['comp_id']}\n{item['n_sites']} sites from {item['ligands']} ligand copies", fontsize=9)
    family_legend(axes[0], th, [f for f in pp.FAMILIES if f != "NegIonizable"], fontsize=8)
    return fig


# -- one compound in every representation of the project -----------------------------------------------------------------------------------------------------

ATLAS_LIGAND = ("cyp3a4", "2V0M", "KLN")  # ketoconazole in CYP3A4, the reference complex of 9.13 and 11.5

XTB_SCRIPT = """
import json
import sys
from cypmode.challenge import physics_block as pb
from cypmode.challenge import quantum_pharmacophore as quantum

smiles, out = sys.argv[1], sys.argv[2]
features, atom_rows = pb.compute_molecule(smiles)
payload = {"physics": features, "atoms": atom_rows, "quantum": quantum.extract_quantum_features(smiles)}
with open(out, "w") as handle:
    json.dump(payload, handle, default=lambda value: value.item() if hasattr(value, "item") else str(value))
"""


def xtb_in_a_process_of_its_own(smiles: str, timeout: int = 900) -> dict:
    """The GFN2-xTB quantities of a compound ({physics, atoms, quantum}), computed by tblite in a fresh interpreter. tblite and LightGBM crash or deadlock when both are loaded in one process (the ``tblite`` marker of
    ``pyproject.toml``), and the test suite, the figure build and the notebook build each load LightGBM somewhere, so the call is never made in the process that draws the figure."""
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / "xtb.json"
        done = subprocess.run([sys.executable, "-B", "-c", XTB_SCRIPT, smiles, str(out)], capture_output=True, text=True, timeout=timeout, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)}, check=False)
        if done.returncode != 0 or not out.exists():
            raise RuntimeError(f"the xTB calculation of the atlas compound failed (exit {done.returncode}): {done.stderr[-800:]}")
        return json.loads(out.read_text())


@functools.cache
def atlas_compound() -> dict:
    """Everything the atlas shows of the compound, computed once: its structure, its physics (per atom and per molecule), its quantum scalars and the hypothesis of the ligand-based model that it realises best."""
    from cypmode.challenge import pharmacophore_hypotheses as hypotheses
    from cypmode.challenge import physics_block as pb

    record = reference_record("cyp3a4")
    smiles = record["smiles"]
    mol = pb.largest_fragment(smiles)
    xtb = xtb_in_a_process_of_its_own(smiles)
    features, atom_rows, scalars = xtb["physics"], xtb["atoms"], xtb["quantum"]
    model, vocabulary, _, _, _, _ = hypotheses.fitted("CYP3A4")
    realised = {int(v) for v in hypotheses.realised(smiles)}
    located, ratio = None, float("nan")
    for k in range(model.n_positive):
        signature = int(vocabulary[model.index[k]])
        if signature in realised:
            located, ratio = hypotheses.locate(smiles, signature), float(model.ratio[k])
            break
    return {"record": record, "smiles": smiles, "mol": mol, "physics": features, "atoms": pd.DataFrame(atom_rows), "quantum": scalars, "located": located, "ratio": ratio}


def colour_map(values: dict[int, float], cmap_name: str, symmetric: bool = False):
    """({atom: rgb}, the ScalarMappable that carries the scale) for values by atom."""
    import matplotlib as mpl

    low, high = (-max(abs(v) for v in values.values()), max(abs(v) for v in values.values())) if symmetric else (min(values.values()), max(values.values()))
    norm = mpl.colors.Normalize(vmin=low, vmax=high if high > low else low + 1)
    cmap = mpl.colormaps[cmap_name]
    return {atom: tuple(cmap(norm(v))[:3]) for atom, v in values.items()}, mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


def atlas_caption() -> str:
    item = atlas_compound()
    hypothesis = ("none of the positive hypotheses of the ligand-based model is realised by it" if item["located"] is None else
                  f"the strongest positive hypothesis of the ligand-based model that it realises ({' + '.join(f['name'] for f in item['located']['features'])}, log-likelihood ratio {item['ratio']:+.2f})")
    return ("One compound, every pharmacophore representation of the project: ketoconazole, the CYP3A4 reference complex 2V0M. (a) Its functional groups (fixed table). (b) Its pharmacophore features by family, the atoms that the "
            "feature-based blocks work with. (c) On a 3D conformer (ETKDG, seed 42: a low-energy shape, not the bound pose), " + hypothesis + ": the translucent spheres are its features, the dashed lines carry the distances "
            "in Å (the three-point pharmacophore of 3.5 and 7.8). (d) The pharmacophore level of the physics block (10.6): the carbon atoms coloured by the energy needed to abstract one of their hydrogens (red the "
            "easiest, blue the hardest: a likely site of oxidation is red) and the heme-binding nitrogen in green; the mean distance between the binder and the easiest site over the conformers is in the label. (e) The electronic level: "
            "the GFN2-xTB partial charge of each atom (blue positive, red negative) with the scalars of the quantum-pharmacophore block (energies in Hartree). (f) The pocket block and the site model in the real crystal structure: the ligand in "
            "the pocket of CYP3A4 with the sites of 9.14 as translucent spheres.")


@figure("pharm_atlas", GROUP, "One compound in every pharmacophore representation of the project", atlas_caption,
        data=(TRAIN, "data/tdi_track/outputs/caches/pharmacophore_hypotheses.npz", "data/tdi_track/outputs/caches/pocket_fields_cyp3a4.npz"), section="9.15")
def atlas(theme_name):
    th = apply(theme_name)
    item = atlas_compound()
    mol = item["mol"]
    fig = plt.figure(figsize=(15.0, 9.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    image, legend = draw_groups(mol, theme_name, size=(560, 420))
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(image)
    blank(ax)
    group_legend(ax, th, legend)
    ax.set_title("(a) functional groups", fontsize=10, loc="left")
    image, families = draw_families(mol, theme_name, size=(560, 420))
    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(image)
    blank(ax)
    family_legend(ax, th, families)
    ax.set_title("(b) pharmacophore features by family", fontsize=10, loc="left")
    ax = fig.add_subplot(grid[0, 2], projection="3d")
    if item["located"] is not None:
        draw_conformer(ax, item["located"], th)
        ax.set_title("(c) the strongest positive hypothesis it realises, in 3D", fontsize=10, loc="left")
    else:
        ax.set_axis_off()
        ax.set_title("(c) no positive hypothesis is realised", fontsize=10, loc="left")
    atoms = item["atoms"]
    hat = {int(r.atom): float(r.hat_kcal) for r in atoms.itertuples() if np.isfinite(r.hat_kcal)}
    colours, mappable = colour_map(hat, "coolwarm_r")
    binders = {int(r.atom) for r in atoms.itertuples() if isinstance(r.binder_role, str) and r.binder_role}
    for atom in binders:
        colours[atom] = hex_to_unit(th["green"])
    ax = fig.add_subplot(grid[1, 0])
    ax.imshow(draw(mol, theme_name, atoms=list(colours), atom_colors=colours, size=(560, 420), line_width=1.8))
    blank(ax)
    fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.01, label="H-abstraction energy (kcal/mol)")
    best = min(hat, key=hat.get) if hat else None
    reach = item["physics"].get("phys_d_binder_best_mean", float("nan"))
    ax.set_title(f"(d) physics: easiest hydrogen abstraction (carbon {best}), heme binder (green)\nmean binder-to-site distance {reach:.1f} Å" if np.isfinite(reach) else "(d) physics: easiest hydrogen abstraction, heme binder (green)", fontsize=9, loc="left")
    charges = {int(r.atom): float(r.q) for r in atoms.itertuples() if np.isfinite(r.q)}
    colours, mappable = colour_map(charges, "coolwarm_r", symmetric=True)
    ax = fig.add_subplot(grid[1, 1])
    ax.imshow(draw(mol, theme_name, atoms=list(colours), atom_colors=colours, size=(560, 420), line_width=1.8))
    blank(ax)
    fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.01, label="partial charge (e)")
    q = item["quantum"]
    ax.set_title("(e) electronic: partial charge of each atom", fontsize=9, loc="left")
    ax.set_xlabel(f"HOMO {q['qm_homo_energy']:.2f} Ha, LUMO {q['qm_lumo_energy']:.2f} Ha, gap {q['qm_homo_lumo_gap']:.2f} Ha, dipole {q['qm_dipole_magnitude']:.2f} (as the block stores it)", fontsize=7)
    ax = fig.add_subplot(grid[1, 2])
    ax.imshow(pocket_images(theme_name)["cyp3a4"]["image"])
    blank(ax)
    family_legend(ax, th, [f for f in pp.FAMILIES if f != "NegIonizable"], fontsize=7)
    ax.set_title("(f) the pocket and the site model, in the crystal", fontsize=10, loc="left")
    return fig
