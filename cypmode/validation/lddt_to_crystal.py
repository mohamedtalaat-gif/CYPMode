"""LDDT-PLI of a predicted pose against the experimental pose of the same ligand, the metric the Structure Track is scored by.

The official score is computed with OpenStructure. This is the same measure as ``pose_selection.lddt_pli_arrays`` made symmetry-aware (the ligand atoms are mapped
through the automorphisms of the SMILES graph, ``pose_pools.symmetric_lddt_pli``) and applied to the crystal: for the protein atoms within 6 A of the crystal ligand,
how well are the distances to the ligand atoms kept, at thresholds of 0.5, 1, 2 and 4 A. It needs no superposition.

Several crystal entries or chains can hold the same ligand and may sit differently, so, like the RMSD to the crystal (``pose_rescoring_validation.true_ligand_rmsd``),
the score of a pose is the best over the crystal copies. Residues are matched by number: the Boltz-2 inputs use the UniProt numbering of CYP3A4 and so do the PDB entries.
The mmCIF files come from ``data/structure_track/inputs/pdb_cache`` (regenerable, see ``reference_ligands``).
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from cypmode.validation.pose_pools import (
    symmetric_lddt_pli,
    template_automorphisms,
    template_ligand_coordinates,
)
from cypmode.validation.pose_rescoring import connectivity_key
from cypmode.validation.pose_rescoring_validation import crystal_mol
from cypmode.validation.pose_selection import complex_arrays
from cypmode.validation.reference_ligands import CACHE_DIR, load_reference_set

COORDINATE_TOLERANCE_A = 0.02  # the ligand atoms of a reference record are found in the mmCIF by their coordinates


@dataclass(frozen=True)
class CrystalReference:
    pdb_id: str
    ligand: np.ndarray  # heavy atoms ordered by the atoms of the SMILES template, shape (K, 3)
    protein: dict  # (residue number, atom name) -> coordinates of the protein chain that holds the ligand


@cache
def _chain_atoms(pdb_id: str, anchor: tuple[float, float, float]) -> dict:
    """Protein heavy atoms, keyed by (residue number, atom name), of the chain of the entry that contains the atom at ``anchor``."""
    from Bio.PDB import MMCIFParser

    from cypmode.validation.structures import _STANDARD_RESIDUES

    structure = MMCIFParser(QUIET=True).get_structure(pdb_id, str(Path(CACHE_DIR) / f"{pdb_id}.cif"))
    point = np.asarray(anchor)
    for chain in structure[0]:
        if any(np.linalg.norm(atom.coord - point) < COORDINATE_TOLERANCE_A for residue in chain for atom in residue):
            return {
                (residue.id[1], atom.get_name()): np.asarray(atom.coord, dtype=float)
                for residue in chain
                if residue.resname in _STANDARD_RESIDUES
                for atom in residue
                if atom.element.strip().upper() not in ("H", "D")
            }
    raise ValueError(f"{pdb_id}: no chain holds the reference ligand")


def crystal_references(smiles: str, records: list[dict] | None = None) -> list[CrystalReference]:
    """One reference per crystal copy of the ligand that matches its SMILES template (a copy with missing or modified atoms is skipped, never forced)."""
    from rdkit import Chem

    template = Chem.MolFromSmiles(smiles)
    key = connectivity_key(smiles)
    out = []
    for record in records if records is not None else load_reference_set():
        if not key or connectivity_key(record["smiles"]) != key:
            continue
        coords = np.asarray(record["coords"], dtype=float)
        mol = crystal_mol(smiles, record["elements"], coords)
        if mol is None:
            continue
        match = mol.GetSubstructMatch(template)
        if len(match) != template.GetNumAtoms():
            continue
        conformer = mol.GetConformer()
        ligand = np.array([list(conformer.GetAtomPosition(i)) for i in match], dtype=float)
        try:
            protein = _chain_atoms(record["pdb_id"], tuple(coords[0]))
        except (ValueError, FileNotFoundError):
            continue
        out.append(CrystalReference(record["pdb_id"], ligand, protein))
    return out


def lddt_against_references(model_ligand: np.ndarray, model_protein: dict, references: list[CrystalReference], permutations: np.ndarray) -> float | None:
    """The best LDDT-PLI over the crystal copies; None when there is no copy to compare with."""
    scores = [symmetric_lddt_pli(model_ligand, model_protein, ref.ligand, ref.protein, permutations) for ref in references]
    return max(scores) if scores else None


def pose_lddt_to_crystal(structure, smiles: str, references: list[CrystalReference], permutations: np.ndarray | None = None) -> float | None:
    """LDDT-PLI of a predicted complex against the crystal; None when the predicted ligand does not match its template or there is no reference."""
    if not references:
        return None
    try:
        ligand = template_ligand_coordinates(structure, smiles)
    except ValueError:
        return None
    permutations = template_automorphisms(smiles) if permutations is None else permutations
    return lddt_against_references(ligand, complex_arrays(structure)[1], references, permutations)
