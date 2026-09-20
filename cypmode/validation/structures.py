"""Check the motif screen against Boltz-2 structure predictions.

cypmode/metrics/motifs.py flags a compound as a candidate Type II inhibitor
from 2D substructure alone. This module checks that call against a 3D
prediction: it parses a Boltz-2-predicted complex where the heme cofactor
is explicitly present (as a `HEM` ligand chain) and covalently bonded to the
real axial cysteine, and measures the distance from the heme iron to the
closest ligand nitrogen. A distance in dative-bond range is what a genuine
Type II coordination pose would look like; the input places no constraint
on the inhibitor's position relative to the heme, so that distance is an
unforced structural prediction, not something built into the input.

This is restricted to the closest ligand nitrogen, not specifically the one
matched by a motif pattern -- mapping a 2D SMARTS match to a 3D atom in the
predicted structure isn't attempted here. For the four reference compounds
this project validates against, the coordinating heterocycle sits at the
core of each scaffold with no other nitrogen nearby, so "closest ligand N"
and "the motif nitrogen" coincide in practice, but that's a property of
these specific molecules, not something this code checks for in general.
"""

import json
from pathlib import Path

from Bio.PDB.MMCIFParser import MMCIFParser

HEME_RESNAME = "HEM"
HEME_IRON_ATOM = "FE"

# Dative Fe-N coordination bonds in solved Type II CYP structures run
# ~2.0-2.3 A (Poulos & Johnson, ch. 3 in Ortiz de Montellano (ed.),
# Cytochrome P450: Structure, Mechanism, and Biochemistry, 3rd ed., 2005).
# A margin above that range distinguishes a coordinating pose from one
# where the ligand is merely nearby in the pocket.
COORDINATION_DISTANCE_CUTOFF_A = 2.6

_STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def load_structure(cif_path: str | Path):
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure("complex", str(cif_path))


def find_heme_iron(structure):
    for atom in structure.get_atoms():
        if atom.get_parent().get_resname() == HEME_RESNAME and atom.get_name() == HEME_IRON_ATOM:
            return atom
    raise ValueError(
        f"no {HEME_IRON_ATOM} atom found in a {HEME_RESNAME} residue -- "
        "was heme included in this prediction's input YAML?"
    )


def ligand_nitrogen_distances(structure, fe_atom) -> list[tuple[str, float]]:
    """Distance (Å) from fe_atom to every nitrogen on the non-protein, non-heme chain."""
    distances = []
    for atom in structure.get_atoms():
        resname = atom.get_parent().get_resname()
        if resname in _STANDARD_RESIDUES or resname == HEME_RESNAME:
            continue
        if atom.element != "N":
            continue
        distances.append((atom.get_name(), float(atom - fe_atom)))
    return sorted(distances, key=lambda pair: pair[1])


def closest_ligand_nitrogen(structure) -> tuple[str, float]:
    fe_atom = find_heme_iron(structure)
    distances = ligand_nitrogen_distances(structure, fe_atom)
    if not distances:
        raise ValueError("no ligand nitrogen atoms found -- is the inhibitor chain present?")
    return distances[0]


def load_affinity(predictions_dir: str | Path) -> dict:
    files = list(Path(predictions_dir).glob("affinity_*.json"))
    if not files:
        raise FileNotFoundError(f"no affinity_*.json in {predictions_dir}")
    return json.loads(files[0].read_text())


def load_confidence(predictions_dir: str | Path) -> dict:
    files = list(Path(predictions_dir).glob("confidence_*model_0.json"))
    if not files:
        raise FileNotFoundError(f"no confidence_*model_0.json in {predictions_dir}")
    return json.loads(files[0].read_text())


def summarize_compound(result_dir: str | Path) -> dict:
    """Summarize one Boltz-2 result directory, e.g. .../boltz_results_cyp3a4_ketoconazole."""
    result_dir = Path(result_dir)
    name = result_dir.name.removeprefix("boltz_results_")
    predictions_dir = result_dir / "predictions" / name
    cif_path = predictions_dir / f"{name}_model_0.cif"

    structure = load_structure(cif_path)
    atom_name, distance = closest_ligand_nitrogen(structure)
    affinity = load_affinity(predictions_dir)
    confidence = load_confidence(predictions_dir)

    return {
        "compound": name,
        "closest_ligand_n_atom": atom_name,
        "fe_n_distance_angstrom": round(distance, 2),
        "coordinated": distance <= COORDINATION_DISTANCE_CUTOFF_A,
        "affinity_pred_value": affinity["affinity_pred_value"],
        "affinity_probability_binary": affinity["affinity_probability_binary"],
        "confidence_score": confidence["confidence_score"],
    }
