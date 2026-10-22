# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Figures about turning a molecule into numbers: the Morgan algorithm step by step on a real compound, Tanimoto similarity between real compounds, and the size of every feature block."""

import io
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import figure
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from style import THEMES, apply, tidy

GROUP = "molecules"
ROOT = Path(__file__).resolve().parents[3]
TRAIN = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"
RADIUS, BITS = 2, 2048


def hex_to_unit(color: str):
    return tuple(int(color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))


def draw(mol, theme_name, atoms=(), bonds=(), size=(520, 400), atom_colors=None, labels=False, line_width: float = 2.0):
    """A depiction of the molecule as an RGB array on the theme's background, with the given atoms and bonds highlighted."""
    theme = THEMES[theme_name]
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    options = drawer.drawOptions()
    if theme_name == "dark":
        rdMolDraw2D.SetDarkMode(options)
    options.setBackgroundColour(hex_to_unit(theme["background"]))
    options.addAtomIndices = labels
    options.bondLineWidth = line_width
    highlight = hex_to_unit(theme["orange"])
    drawer.DrawMolecule(mol, highlightAtoms=list(atoms), highlightBonds=list(bonds),
                        highlightAtomColors={a: (atom_colors or {}).get(a, highlight) for a in atoms}, highlightBondColors={b: highlight for b in bonds})
    drawer.FinishDrawing()
    return np.asarray(Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB"))


@cache
def labelled_compounds() -> pd.DataFrame:
    train = pd.read_csv(ROOT / TRAIN)
    keep = train["CYP3A4_is_TDI"].notna() & train["CYP3A4_pIC50_TDI_condition"].notna() & train["CYP3A4_pIC50_direct_inhibition"].notna()
    out = train.loc[keep, ["Molecule_Name", "SMILES", "CYP3A4_is_TDI"]].reset_index(drop=True)
    out["mol"] = [Chem.MolFromSmiles(s) for s in out["SMILES"]]
    out["heavy"] = [m.GetNumHeavyAtoms() for m in out["mol"]]
    return out


def example_compound() -> pd.Series:
    """The first time-dependent inhibitor of the training file with 18 to 22 heavy atoms that has a nitrogen and an aromatic ring: small enough to draw, rich enough to show the algorithm."""
    table = labelled_compounds()
    nitrogen = Chem.MolFromSmarts("[#7]")
    aromatic = Chem.MolFromSmarts("a")
    for row in table.itertuples():
        if row.CYP3A4_is_TDI and 18 <= row.heavy <= 22 and row.mol.HasSubstructMatch(nitrogen) and row.mol.HasSubstructMatch(aromatic):
            return table.iloc[row.Index]
    raise RuntimeError("no example compound")


def morgan(mol):
    """The fingerprint and, for every bit switched on, the (atom, radius) environments that hash to it."""
    generator = AllChem.GetMorganGenerator(radius=RADIUS, fpSize=BITS)
    output = AllChem.AdditionalOutput()
    output.AllocateBitInfoMap()
    fp = generator.GetFingerprint(mol, additionalOutput=output)
    return fp, {bit: list(environments) for bit, environments in output.GetBitInfoMap().items()}


def morgan_caption() -> str:
    compound = example_compound()
    fp, info = morgan(compound["mol"])
    by_radius = [sum(1 for envs in info.values() for _, r in envs if r == k) for k in range(RADIUS + 1)]
    return (f"The Morgan algorithm on one real training compound ({compound['heavy']} heavy atoms; its identifier is withheld because the compound is in the blinded set). Radius 0 looks at each atom alone, radius 1 adds its bonded neighbours, "
            f"radius 2 the neighbours of the neighbours; the highlighted atoms are the environment of one atom at each radius. Every environment is hashed to a number between 0 and {BITS - 1} and "
            f"that position of the fingerprint is switched on. This compound has {by_radius[0]} environments of radius 0, {by_radius[1]} of radius 1 and {by_radius[2]} of radius 2, "
            f"which switch on {fp.GetNumOnBits()} of the {BITS} bits (two environments that hash to the same position share a bit).")


@figure("mol_morgan_steps", GROUP, "The Morgan algorithm, radius by radius", morgan_caption, data=(TRAIN,), section="3.2")
def morgan_steps(theme_name):
    th = apply(theme_name)
    compound = example_compound()
    mol = compound["mol"]
    centre = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetDegree() >= 2)
    fig, axes = plt.subplots(1, RADIUS + 1, figsize=(12, 4.4))
    for r, ax in enumerate(axes):
        atoms, bonds = set(), []
        if r == 0:
            atoms = {centre}
        else:
            bond_ids = Chem.FindAtomEnvironmentOfRadiusN(mol, r, centre)
            bonds = list(bond_ids)
            atoms = {centre} | {a for b in bond_ids for a in (mol.GetBondWithIdx(b).GetBeginAtomIdx(), mol.GetBondWithIdx(b).GetEndAtomIdx())}
        ax.imshow(draw(mol, theme_name, atoms, bonds))
        ax.set_title(f"radius {r}: {len(atoms)} atom{'s' if len(atoms) != 1 else ''} around the nitrogen", fontsize=10, color=th["text"])
        ax.axis("off")
    fig.tight_layout()
    return fig


def similar_pair():
    """The example compound, its nearest neighbour by Tanimoto similarity and a compound at the median similarity."""
    table = labelled_compounds()
    compound = example_compound()
    generator = AllChem.GetMorganGenerator(radius=RADIUS, fpSize=BITS)
    fps = [generator.GetFingerprint(m) for m in table["mol"]]
    query = generator.GetFingerprint(compound["mol"])
    sims = np.array(DataStructs.BulkTanimotoSimilarity(query, fps))
    sims[table["Molecule_Name"] == compound["Molecule_Name"]] = -1
    nearest = int(sims.argmax())
    median = int(np.argsort(sims)[len(sims) // 2])
    return query, [(table.iloc[nearest], generator.GetFingerprint(table.iloc[nearest]["mol"]), sims[nearest]), (table.iloc[median], generator.GetFingerprint(table.iloc[median]["mol"]), sims[median])], sims


def tanimoto_caption() -> str:
    _, others, sims = similar_pair()
    return (f"Tanimoto similarity T = |A and B| / |A or B| between the example compound and two other training compounds: its nearest neighbour (T = {others[0][2]:.2f}) and a compound at the "
            f"median similarity (T = {others[1][2]:.2f}). The bars count the fingerprint bits only the example has, the bits both have, and the bits only the other has. Over all "
            f"{len(sims) - 1} other compounds the median similarity is {np.median(sims[sims >= 0]):.2f}, and {int((sims >= 0.5).sum())} of them reach 0.5 or more.")


@figure("mol_tanimoto", GROUP, "Tanimoto similarity between real compounds", tanimoto_caption, data=(TRAIN,), section="3.4")
def tanimoto(theme_name):
    th = apply(theme_name)
    query, others, _ = similar_pair()
    compound = example_compound()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), gridspec_kw={"height_ratios": [1.3, 1]})
    a_bits = set(query.GetOnBits())
    for col, (other, fp, sim) in enumerate(others):
        b_bits = set(fp.GetOnBits())
        axes[0][col].imshow(draw(other["mol"], theme_name, size=(420, 320)))
        axes[0][col].set_title(f"{other['Molecule_Name']}  (T = {sim:.2f})", fontsize=10)
        axes[0][col].axis("off")
        ax = axes[1][col]
        counts = [len(a_bits - b_bits), len(a_bits & b_bits), len(b_bits - a_bits)]
        ax.bar(["only the\nexample", "both", "only the\nother"], counts, color=[th["blue"], th["green"], th["red"]])
        ax.bar_label(ax.containers[0], padding=2, fontsize=9)
        ax.set(ylabel="bits switched on" if col == 0 else "")
        tidy(ax, th, grid=False)
    axes[0][2].imshow(draw(compound["mol"], theme_name, size=(420, 320)))
    axes[0][2].set_title(f"example: {compound['Molecule_Name']}", fontsize=10)
    axes[0][2].axis("off")
    axes[1][2].axis("off")
    fig.tight_layout()
    return fig


def block_sizes() -> pd.Series:
    store = np.load(ROOT / STORE, allow_pickle=True)
    return pd.Series({name: store[name].shape[1] for name in store.files if name != "smiles"}).sort_values()


def blocks_caption() -> str:
    sizes = block_sizes()
    return (f"Width of every feature block of the feature store, in columns (log scale), for the {len(np.load(ROOT / STORE, allow_pickle=True)['smiles'])} compounds of the training and test files together. "
            f"The fingerprints are the wide blocks ({sizes.idxmax()} has {sizes.max()} columns); the physics, alert and motif blocks are narrow but carry chemistry the fingerprints cannot. "
            f"Together the {len(sizes)} blocks hold {int(sizes.sum())} columns.")


@figure("mol_feature_blocks", GROUP, "The feature blocks and their sizes", blocks_caption, data=(STORE,), section="3.7")
def feature_blocks(theme_name):
    th = apply(theme_name)
    sizes = block_sizes()
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.barh(sizes.index, sizes.to_numpy(), color=th["blue"])
    ax.bar_label(ax.containers[0], padding=3, fontsize=8)
    ax.set_xscale("log")
    ax.set(xlabel="columns (log scale)", xlim=(0.8, sizes.max() * 2))
    tidy(ax, th, grid=False)
    fig.tight_layout()
    return fig
