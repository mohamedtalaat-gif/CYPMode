"""The heme pocket of CYP3A4 rendered from real crystal structures: a Type II inhibitor with its nitrogen on the iron, and a ligand that fills the pocket without touching the iron.

The pictures are ray-traced by PyMOL (``gen/pymol_scene.py``) from the deposited coordinates (``data/structure_track/inputs/pdb_cache``) and cut open through the heme plane; the labels are drawn on top
at the projected positions of the atoms, and the distances in the caption are measured on the coordinates.
"""

import json
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import MMCIFParser
from catalog import figure
from style import apply

from gen.pymol_scene import pocket as render_pocket

GROUP = "biology"
ROOT = Path(__file__).resolve().parents[3]
REFERENCES = "data/structure_track/inputs/cyp3a4_reference_ligands.json"
CACHE = ROOT / "data/structure_track/inputs/pdb_cache"
AXIAL_CYSTEINE = 442
SHELL_A = 4.0  # a residue is part of the pocket when one of its atoms is this close to the ligand
HBOND_A, HYDROPHOBIC_A, COORDINATION_A = 3.4, 4.0, 2.6
EXAMPLES = (("6MA7", "TPF", "Type II: the nitrogen binds the iron"), ("6OOB", "MWS", "Type I-like: ligand above the heme, no contact with the iron"))
THREE = {"ALA": "Ala", "ARG": "Arg", "ASN": "Asn", "ASP": "Asp", "CYS": "Cys", "GLN": "Gln", "GLU": "Glu", "GLY": "Gly", "HIS": "His", "ILE": "Ile", "LEU": "Leu", "LYS": "Lys",
         "MET": "Met", "PHE": "Phe", "PRO": "Pro", "SER": "Ser", "THR": "Thr", "TRP": "Trp", "TYR": "Tyr", "VAL": "Val"}
BACKBONE = {"N", "C", "O", "CA"}


class Atom:
    def __init__(self, element, name, xyz, group, residue=None):
        self.element, self.name, self.xyz, self.group, self.residue = element.upper(), name, np.asarray(xyz, dtype=float), group, residue


@cache
def reference_record(pdb_id: str, comp_id: str) -> dict:
    records = json.loads((ROOT / REFERENCES).read_text())["ligands"]
    return min((r for r in records if r["pdb_id"] == pdb_id and r["comp_id"] == comp_id), key=lambda r: r["min_fe_distance"])


@cache
def pocket_atoms(pdb_id: str, comp_id: str):
    """(atoms, residue labels, entry resolution): ligand, heme, the pocket residues and any water on the iron, read from the deposited entry."""
    record = reference_record(pdb_id, comp_id)
    structure = MMCIFParser(QUIET=True).get_structure(pdb_id, str(CACHE / f"{pdb_id}.cif"))
    anchor = np.asarray(record["coords"][0], dtype=float)
    chain = next(c for c in structure[0] if any(np.linalg.norm(a.coord - anchor) < 0.02 for r in c for a in r))
    ligand = [Atom(e, f"{e}{i}", xyz, "ligand") for i, (e, xyz) in enumerate(zip(record["elements"], record["coords"], strict=True))]
    centre = np.mean([a.xyz for a in ligand], axis=0)
    hemes = [r for r in chain if r.resname == "HEM"]
    heme = min(hemes, key=lambda r: np.linalg.norm(np.mean([a.coord for a in r if a.element == "FE"], axis=0) - centre))
    heme_atoms = [Atom(a.element, a.get_name(), a.coord, "heme") for a in heme if a.element.strip().upper() not in ("H", "D")]
    residues = []
    for r in chain:
        if r.resname in THREE:
            atoms = [Atom(a.element, a.get_name(), a.coord, "protein", r) for a in r if a.element.strip().upper() not in ("H", "D")]
            near = min((np.linalg.norm(a.xyz - l.xyz) for a in atoms for l in ligand), default=99)
            if near <= SHELL_A or r.id[1] == AXIAL_CYSTEINE:
                residues.append((near, r, atoms))
    fe = next(a for a in heme_atoms if a.element == "FE").xyz
    waters = [Atom("O", "HOH", a.coord, "water") for r in chain if r.resname == "HOH" for a in r if np.linalg.norm(a.coord - fe) <= COORDINATION_A]
    residues.sort(key=lambda t: t[0])
    pocket = [(r, atoms) for near, r, atoms in residues if r.id[1] != AXIAL_CYSTEINE][:14] + [(r, atoms) for _, r, atoms in residues if r.id[1] == AXIAL_CYSTEINE]
    resolution = record["resolution"]
    return ligand, heme_atoms, pocket, waters, resolution


def interactions(ligand, heme, pocket, waters):
    """(coordination, hydrogen bonds, hydrophobic contacts): ligand atom, partner and distance, measured on the coordinates."""
    fe = next(a for a in heme if a.element == "FE")
    coordination = [(l, fe, float(np.linalg.norm(l.xyz - fe.xyz))) for l in ligand if l.element in ("N", "O") and np.linalg.norm(l.xyz - fe.xyz) <= COORDINATION_A]
    hbonds, hydrophobic = [], {}
    for r, atoms in pocket:
        for a in atoms:
            for l in ligand:
                d = float(np.linalg.norm(a.xyz - l.xyz))
                if a.element in ("N", "O") and l.element in ("N", "O") and d <= HBOND_A and r.id[1] != AXIAL_CYSTEINE:
                    hbonds.append((l, a, d))
                elif a.element == "C" and l.element == "C" and d <= HYDROPHOBIC_A and a.name not in BACKBONE - {"CA"}:
                    hydrophobic[r.id[1]] = hydrophobic.get(r.id[1], 0) + 1
    return coordination, hbonds, hydrophobic



def pocket_caption() -> str:
    parts = []
    for pdb_id, comp_id, kind in EXAMPLES:
        ligand, heme, pocket, waters, resolution = pocket_atoms(pdb_id, comp_id)
        coordination, hbonds, _hydrophobic = interactions(ligand, heme, pocket, waters)
        fe = next(a.xyz for a in heme if a.element == "FE")
        closest = min(float(np.linalg.norm(l.xyz - fe)) for l in ligand)
        iron = (f"a nitrogen of the ligand sits {coordination[0][2]:.2f} A from the iron" if coordination
                else f"the closest ligand atom is {closest:.2f} A from the iron" + (f", and a water molecule {np.linalg.norm(waters[0].xyz - fe):.2f} A from the iron still occupies the sixth position" if waters else ""))
        parts.append(f"{kind.split(':')[0]} (PDB {pdb_id}, {resolution:.2f} A): {iron}; {len(pocket)} residues line the pocket, with {len(hbonds)} polar contact{'' if len(hbonds) == 1 else 's'} within {HBOND_A} A")
    return ("Two real CYP3A4 co-crystal structures, rendered from the deposited coordinates and cut open through the heme plane: the protein as a translucent cartoon, the pocket as a surface, "
            "the lining residues as thin sticks, the heme in orange with its iron as a sphere, the ligand in teal, and the iron coordination as a dashed line. " + "; ".join(parts) + ". Every distance is measured on the coordinates.")


@figure("bio_type1_type2", GROUP, "Two ways to bind the heme, on real structures: Type II and Type I", pocket_caption, data=("data/structure_track/inputs/pdb_cache/6MA7.cif", "data/structure_track/inputs/pdb_cache/6OOB.cif"), section="1.4")
def type_one_two(theme_name):
    th = apply(theme_name)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.9))
    for ax, (pdb_id, comp_id, title) in zip(axes, EXAMPLES, strict=True):
        scene = render_pocket(pdb_id, comp_id, theme=theme_name)
        ax.imshow(plt.imread(scene.png))
        ax.set_xlim(330, 1470)
        ax.set_ylim(930, 190)
        ax.axis("off")
        record = reference_record(pdb_id, comp_id)
        ax.set_title(f"{title}\nPDB {pdb_id}, ligand {comp_id}, resolution {record['resolution']:.2f} A", fontsize=12, color=th["text"], loc="left", fontweight="bold")
        fe, anchors = scene.anchors["fe"], scene.anchors
        ax.annotate("Fe", fe, xytext=(fe[0] - 150, fe[1] + 70), color=th["orange"], fontsize=13, fontweight="bold", arrowprops={"arrowstyle": "-", "color": th["orange"], "lw": 1.6}, ha="center")
        ax.annotate("axial cysteine (S)", anchors["cys"], xytext=(anchors["cys"][0] - 160, anchors["cys"][1] + 110), color=th["gold"], fontsize=11, arrowprops={"arrowstyle": "-", "color": th["gold"], "lw": 1.4}, ha="center")
        ax.annotate("ligand", anchors["ligand"], xytext=(anchors["ligand"][0] + 250, anchors["ligand"][1] - 170), color=th["teal"], fontsize=13, fontweight="bold", arrowprops={"arrowstyle": "-", "color": th["teal"], "lw": 1.6}, ha="center")
        if "coordinating_n" in anchors:
            n = anchors["coordinating_n"]
            ax.text(fe[0] + 85, (n[1] + fe[1]) / 2 - 20, f"{record['min_fe_distance']:.2f} A", color=th["gold"], fontsize=13, fontweight="bold", va="center")
        else:
            ax.text(fe[0] + 230, fe[1] + 270, f"closest ligand atom\n{record['min_fe_distance']:.2f} A from the iron", color=th["gold"], fontsize=11.5, va="center", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
