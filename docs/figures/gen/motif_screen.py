"""The motif screen on four real drugs: the heme-coordinating nitrogen group the screen finds, and whether Boltz-2 puts its nitrogen on the iron of CYP3A4."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from catalog import figure
from rdkit import Chem
from style import apply

from cypmode.metrics.motifs import TYPE_II_MOTIFS
from gen.molecules import draw

GROUP = "biology"
ROOT = Path(__file__).resolve().parents[3]
INPUTS = ROOT / "data/structure_track/inputs"
SUMMARY = "data/structure_track/outputs/cyp3a4/validation_summary.json"
DRUGS = ("ketoconazole", "ritonavir", "azamulin", "vardenafil")


def smiles_of(name: str) -> str:
    spec = yaml.safe_load((INPUTS / f"cyp3a4_{name}.yaml").read_text())
    return next(entry["ligand"]["smiles"] for entry in spec["sequences"] if "ligand" in entry)


def motif_atoms(mol) -> tuple[str, tuple[int, ...]]:
    """The first Type II motif of the screen that matches the molecule, and the atoms of every match of it (a molecule can carry the ring twice)."""
    for name, smarts in TYPE_II_MOTIFS.items():
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        if matches:
            return name, tuple(sorted({atom for match in matches for atom in match}))
    return "", ()


def motif_caption() -> str:
    summary = json.loads((ROOT / SUMMARY).read_text())
    parts = [f"{name} {summary[name]['closest_motif_nitrogen_distance_angstrom']:.2f} A ({'on the iron' if summary[name]['coordinated'] else 'not on the iron'})" for name in DRUGS]
    return ("The motif screen on the four drugs whose structures bound to CYP3A4 were solved by cryo-EM. Each structure is drawn from its SMILES; the atoms of the first heme-coordinating motif the screen finds "
            "(imidazole, triazole, pyridine or thiazole rings, defined in cypmode/metrics/motifs.py) are highlighted. Below each, the distance from the heme iron to the nearest motif nitrogen in the "
            "complex Boltz-2 predicted without any restraint: " + "; ".join(parts) + ". Two drugs put the nitrogen on the iron, two do not, although all four match a motif.")


@figure("bio_motif_screen", GROUP, "The heme-binding motif screen on four real drugs", motif_caption, data=(SUMMARY, "data/structure_track/inputs/cyp3a4_ketoconazole.yaml"), section="5.1")
def motif_screen(theme_name):
    th = apply(theme_name)
    summary = json.loads((ROOT / SUMMARY).read_text())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, name in zip(axes, DRUGS, strict=True):
        mol = Chem.MolFromSmiles(smiles_of(name))
        motif, atoms = motif_atoms(mol)
        ax.imshow(draw(mol, theme_name, atoms=atoms, size=(760, 600), line_width=4))
        ax.axis("off")
        entry = summary[name]
        bound = entry["coordinated"]
        ax.set_title(name, fontsize=13, fontweight="bold", color=th["text"])
        ax.text(0.5, -0.04, f"motif: {motif or 'none'}\niron to nitrogen: {entry['closest_motif_nitrogen_distance_angstrom']:.2f} A",
                transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color=th["green"] if bound else th["red"], fontweight="bold")
    fig.tight_layout()
    return fig
