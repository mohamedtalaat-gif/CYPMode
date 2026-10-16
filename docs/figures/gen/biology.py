"""The biology: the catalytic cycle of a cytochrome P450 around its real heme, and the route of a drug through the body with the real enzyme and a real compound."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import numpy as np
from rdkit import Chem

from catalog import figure
from gen._draw import arrow, box, canvas
from gen.molecules import draw
from gen.pymol_scene import autocrop, heme_closeup, overview
from style import apply

GROUP = "biology"
ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = "OCNT-2395436"  # the first official structure ligand: one real compound is followed through the notebook and the film

CYCLE = [
    ("1  Resting enzyme", "Fe(III) with water as the sixth ligand"),
    ("2  Substrate binds", "RH enters the pocket and pushes the water out"),
    ("3  First electron", "Fe(III) to Fe(II), from NADPH through the reductase"),
    ("4  Oxygen binds", "O2 attaches to Fe(II)"),
    ("5  Second electron", "a peroxo complex, Fe(III)-O-O"),
    ("6  Two protons, water leaves", "Compound I: an iron(IV) oxo with a radical on the porphyrin"),
    ("7  Hydrogen abstraction", "the oxo takes a hydrogen from RH and leaves a radical R"),
    ("8  Oxygen rebound", "the hydroxyl joins R: ROH, which leaves; water returns"),
]


SUBSTRATE = "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"  # testosterone (stereochemistry not drawn)
PRODUCT = "CC12CCC3C(C1CCC2O)CC(O)C4=CC(=O)CCC34C"  # 6-hydroxytestosterone, the classic CYP3A4 product


@figure("bio_p450_cycle", GROUP, "The catalytic cycle of a cytochrome P450",
        "The eight steps by which a P450 oxidises a substrate RH to ROH with one oxygen atom from O2 and two electrons from NADPH. A molecule that competes for step 2, or that binds the iron "
        "so that step 4 cannot happen, switches the enzyme off while it is bound (reversible inhibition). A molecule that the enzyme itself converts in steps 6 to 8 into something that "
        "stays attached to the heme or the protein switches it off for good: that is time-dependent inhibition, the label this project predicts. In the middle, the real heme and the axial cysteine "
        "of CYP3A4 rendered from its crystal structure (PDB 6MA7); at the top, a real substrate of the enzyme, testosterone, and its classic CYP3A4 product, 6-hydroxytestosterone.",
        data=("data/structure_track/inputs/pdb_cache/6MA7.cif",), section="1.3")
def p450_cycle(theme_name):
    th = apply(theme_name)
    fig, ax = plt.subplots(figsize=(10.5, 8.6))
    canvas(ax, (-6.6, 6.6), (-5.6, 6.4))
    n = len(CYCLE)
    centres = [(4.1 * math.sin(2 * math.pi * i / n), 3.9 * math.cos(2 * math.pi * i / n)) for i in range(n)]
    colors = [th["panel"], th["panel"], th["panel"], th["panel"], th["panel"], th["orange"], th["orange"], th["panel"]]
    for (cx, cy), (title, detail), fill in zip(centres, CYCLE, colors, strict=True):
        ax.add_patch(FancyBboxPatch((cx - 1.55, cy - 0.62), 3.1, 1.24, boxstyle="round,pad=0.02,rounding_size=0.15", linewidth=1.6, edgecolor=th["text"], facecolor=fill))
        ax.text(cx, cy + 0.27, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=th["text"] if fill == th["panel"] else "#1A1A1A")
        wrapped = detail if len(detail) < 32 else detail[: detail.rfind(" ", 0, 32)] + "\n" + detail[detail.rfind(" ", 0, 32) + 1 :]
        ax.text(cx, cy - 0.22, wrapped, ha="center", va="center", fontsize=7.4, color=th["text"] if fill == th["panel"] else "#1A1A1A")
    for i in range(n):
        a, b = centres[i], centres[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        arrow(ax, (a[0] + ux * 1.25 + 0.0, a[1] + uy * 0.95), (b[0] - ux * 1.25, b[1] - uy * 0.95), th["text"], rad=0.15)
    labels = {0: "substrate RH in", 1: "e- from NADPH", 2: "O2 in", 3: "e- (and H+)", 4: "2 H+ in, H2O out", 7: "ROH out, H2O in"}
    for k, label in labels.items():
        a, b = centres[k], centres[(k + 1) % n]
        ax.text((a[0] + b[0]) / 2 * 1.42, (a[1] + b[1]) / 2 * 1.42, label, ha="center", va="center", fontsize=8.5, color=th["blue"], style="italic")
    heme = autocrop(plt.imread(heme_closeup("6MA7", "TPF", theme=theme_name).png), 30)
    ax.imshow(heme, extent=(-2.3, 2.3, -0.95, 0.95), aspect="auto", zorder=0)
    ax.text(0, -1.4, "the heme and the axial cysteine of CYP3A4\n(PDB 6MA7); the iron cycles between\nFe(III), Fe(II) and Fe(IV)", ha="center", va="top", fontsize=8.6, color=th["muted"])
    for name, smiles, x in (("substrate RH: testosterone", SUBSTRATE, 5.0), ("product ROH: 6-hydroxytestosterone", PRODUCT, -5.0)):
        image = draw(Chem.MolFromSmiles(smiles), theme_name, size=(720, 520), line_width=6)
        ax.imshow(image, extent=(x - 1.7, x + 1.7, 4.2, 5.4), aspect="auto", zorder=1)
        ax.text(x, 6.0, name, ha="center", va="center", fontsize=8.8, color=th["text"], fontweight="bold")
    fig.tight_layout()
    return fig


STAGES = [
    ("swallowed", "absorption: the drug moves from the gut into the blood"),
    ("blood and tissues", "distribution: it spreads through the body"),
    ("liver", "metabolism: P450 enzymes, mainly CYP3A4, oxidise it"),
    ("kidney and bile", "excretion: the products leave the body"),
]


@figure("bio_adme_route", GROUP, "A drug's route through the body and where this project looks",
        "Absorption, distribution, metabolism and excretion in one line. The project predicts what happens at the third stage: whether a molecule switches off the enzymes that do the metabolism, "
        "and so changes how long other drugs taken at the same time stay in the body.",
        section="1.1")
def adme_route(theme_name):
    th = apply(theme_name)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    canvas(ax, (0, 12), (0, 4.4))
    xs = [1.5, 4.5, 7.5, 10.5]
    for x, (name, text), highlight in zip(xs, STAGES, (False, False, True, False), strict=True):
        box(ax, x, 1.1, 2.5, 1.5, text, th["orange"] if highlight else th["panel"], th["text"], "#1A1A1A" if highlight else th["text"], size=8.4, wrap=26, title=name)
    for a, b in zip(xs, xs[1:], strict=False):
        arrow(ax, (a + 1.3, 1.1), (b - 1.3, 1.1), th["text"])
    ax.text(7.5, 0.18, "what CYPMode predicts", ha="center", fontsize=10, color=th["orange"], fontweight="bold")
    from gen.molecules import draw as draw_molecule  # the compound is drawn like every other molecule of the figures
    import pandas as pd

    smiles = pd.read_csv(ROOT / "data/structure_track/inputs/structure_track_test_blinded.csv").set_index("Molecule_Name").loc[EXAMPLE, "SMILES"]
    ax.imshow(draw_molecule(Chem.MolFromSmiles(smiles), theme_name, size=(780, 560), line_width=6), extent=(0.15, 2.85, 2.5, 4.1), aspect="auto", zorder=1)
    ax.text(1.5, 2.4, f"{EXAMPLE}, a real compound of the challenge", ha="center", va="top", fontsize=7.8, color=th["muted"])
    enzyme = autocrop(plt.imread(overview("6MA7", "TPF", theme=theme_name, size=(1200, 760)).png), 20)
    ax.imshow(enzyme, extent=(8.3, 10.7, 2.5, 4.3), aspect="auto", zorder=1)
    ax.text(9.5, 2.4, "CYP3A4 with its heme (orange), from the crystal structure 6MA7", ha="center", va="top", fontsize=7.8, color=th["muted"])
    fig.tight_layout()
    return fig

