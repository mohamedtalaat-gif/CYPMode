"""Check the motif screen against Boltz-2 structure predictions.

cypmode/metrics/motifs.py flags a compound as a candidate Type II inhibitor
from 2D substructure alone. This module checks that call against a 3D
prediction: it parses a Boltz-2-predicted complex where the heme cofactor
is explicitly present (as a `HEM` ligand chain) and covalently bonded to the
real axial cysteine, and measures the distance from the heme iron to the
ligand nitrogen(s) that actually match a Type II motif -- not just whichever
ligand nitrogen happens to sit closest to the iron. The input places no
constraint on the inhibitor's position relative to the heme, so those
distances are an unforced structural prediction, not something built into
the input.

Earlier versions of this module used "closest ligand nitrogen, regardless of
which one" as a proxy for "the coordinating nitrogen." That's wrong in
general -- confirmed directly on vardenafil, where the geometrically closest
ligand nitrogen (4.84 A) isn't part of any matched Type II motif at all; the
nitrogens the motif screen actually flagged sit at 10.5-11.8 A. The fix
reconstructs the ligand's real bonds and aromaticity from its SMILES
(RDKit's AssignBondOrdersFromTemplate, the same technique
github.com/PatWalters/cyp-heatmap uses) so the same RDKit atom indices used
for SMARTS matching also carry the predicted 3D coordinates -- no separate,
fragile mapping between a 2D match and a 3D atom name.
"""

from pathlib import Path

from Bio.PDB.MMCIFParser import MMCIFParser
from rdkit import Chem
from rdkit.Chem import AllChem

from cypmode.metrics.motifs import TYPE_II_MOTIFS
from cypmode.validation.constants import COORDINATION_DISTANCE_CUTOFF_A

HEME_RESNAME = "HEM"
HEME_IRON_ATOM = "FE"

__all__ = [
    "COORDINATION_DISTANCE_CUTOFF_A",
    "all_ligand_nitrogen_distances",
    "find_heme_iron",
    "load_affinity",
    "load_confidence",
    "load_structure",
    "motif_nitrogen_distances",
    "summarize_compound",
]

_STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

_COMPILED_MOTIFS = {name: Chem.MolFromSmarts(smarts) for name, smarts in TYPE_II_MOTIFS.items()}


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


def _ligand_atoms(structure):
    """Non-protein, non-heme atoms -- the predicted inhibitor, whatever it is."""
    return [
        atom
        for atom in structure.get_atoms()
        if atom.get_parent().get_resname() not in _STANDARD_RESIDUES
        and atom.get_parent().get_resname() != HEME_RESNAME
    ]


def _ligand_mol_3d(structure, ligand_smiles: str):
    """Rebuild the ligand as an RDKit mol with correct bonds/aromaticity and a
    3D conformer from the predicted coordinates, so SMARTS-matched atom
    indices and 3D positions refer to the same atoms.

    Returns (mol, pdb_atom_names) -- pdb_atom_names[i] is the CIF atom name
    for mol atom i, for reporting.
    """
    ligand_atoms = _ligand_atoms(structure)
    if not ligand_atoms:
        raise ValueError("no ligand atoms found -- is the inhibitor chain present?")

    lines = []
    for i, atom in enumerate(ligand_atoms, start=1):
        x, y, z = atom.get_coord()
        lines.append(
            f"HETATM{i:>5} {atom.get_name():<4} LIG A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom.element:>2}"
        )
    lines.append("END")
    raw_mol = Chem.MolFromPDBBlock("\n".join(lines), sanitize=False, proximityBonding=True)
    if raw_mol is None:
        raise ValueError("could not parse predicted ligand geometry as a molecule")

    template = Chem.MolFromSmiles(ligand_smiles)
    if template is None:
        raise ValueError(f"could not parse ligand_smiles: {ligand_smiles!r}")

    mol = AllChem.AssignBondOrdersFromTemplate(template, raw_mol)
    pdb_atom_names = [atom.get_name() for atom in ligand_atoms]
    return mol, pdb_atom_names


def motif_nitrogen_distances(structure, ligand_smiles: str, fe_atom) -> list[dict]:
    """Distance (A) from fe_atom to every ligand nitrogen that matches a Type
    II motif, tagged with which motif matched. Sorted closest first.
    """
    mol, pdb_atom_names = _ligand_mol_3d(structure, ligand_smiles)
    conf = mol.GetConformer()

    results = []
    for motif_name, pattern in _COMPILED_MOTIFS.items():
        for match in mol.GetSubstructMatches(pattern):
            for atom_idx in match:
                if mol.GetAtomWithIdx(atom_idx).GetSymbol() != "N":
                    continue
                pos = conf.GetAtomPosition(atom_idx)
                dx = pos.x - float(fe_atom.coord[0])
                dy = pos.y - float(fe_atom.coord[1])
                dz = pos.z - float(fe_atom.coord[2])
                distance = (dx**2 + dy**2 + dz**2) ** 0.5
                results.append(
                    {"atom": pdb_atom_names[atom_idx], "motif": motif_name, "distance_angstrom": round(distance, 2)}
                )
    return sorted(results, key=lambda r: r["distance_angstrom"])


def all_ligand_nitrogen_distances(structure, fe_atom) -> list[dict]:
    """Distance (A) from fe_atom to every ligand nitrogen, motif-matched or
    not -- for manual review, so nothing is silently hidden behind the
    motif-based filter above.
    """
    results = []
    for atom in _ligand_atoms(structure):
        if atom.element != "N":
            continue
        results.append({"atom": atom.get_name(), "distance_angstrom": round(float(atom - fe_atom), 2)})
    return sorted(results, key=lambda r: r["distance_angstrom"])


def load_affinity(predictions_dir: str | Path) -> dict:
    files = list(Path(predictions_dir).glob("affinity_*.json"))
    if not files:
        raise FileNotFoundError(f"no affinity_*.json in {predictions_dir}")
    import json

    return json.loads(files[0].read_text())


def load_confidence(predictions_dir: str | Path) -> dict:
    files = list(Path(predictions_dir).glob("confidence_*model_0.json"))
    if not files:
        raise FileNotFoundError(f"no confidence_*model_0.json in {predictions_dir}")
    import json

    return json.loads(files[0].read_text())


def summarize_compound(result_dir: str | Path, ligand_smiles: str) -> dict:
    """Summarize one Boltz-2 result directory, e.g. .../boltz_results_cyp3a4_ketoconazole.

    ligand_smiles is required -- motif matching needs the 2D structure, not
    just the predicted geometry, and this function doesn't assume or look up
    a fixed set of known compounds.
    """
    result_dir = Path(result_dir)
    name = result_dir.name.removeprefix("boltz_results_")
    predictions_dir = result_dir / "predictions" / name
    cif_path = predictions_dir / f"{name}_model_0.cif"

    structure = load_structure(cif_path)
    fe_atom = find_heme_iron(structure)
    motif_distances = motif_nitrogen_distances(structure, ligand_smiles, fe_atom)
    all_distances = all_ligand_nitrogen_distances(structure, fe_atom)

    closest_motif_distance = motif_distances[0]["distance_angstrom"] if motif_distances else None
    affinity = load_affinity(predictions_dir)
    confidence = load_confidence(predictions_dir)

    return {
        "compound": name,
        "motif_nitrogen_distances": motif_distances,
        "all_ligand_nitrogen_distances": all_distances,
        "closest_motif_nitrogen_distance_angstrom": closest_motif_distance,
        "coordinated": closest_motif_distance is not None and closest_motif_distance <= COORDINATION_DISTANCE_CUTOFF_A,
        "affinity_pred_value": affinity["affinity_pred_value"],
        "affinity_probability_binary": affinity["affinity_probability_binary"],
        "confidence_score": confidence["confidence_score"],
    }
