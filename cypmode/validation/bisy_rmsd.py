"""A local approximation of BiSyRMSD, the Structure Track's own secondary
metric (see CYPMode_Architecture_Specification.md's "Official
challenge update" section) -- for internal self-consistency checks, NOT
exact leaderboard parity. The real scoring tool, OpenStructure, is real
and published but ships Docker/conda-only (no pip wheel) -- confirmed
directly, and declined for local install as too heavy for what this
project needs it for (see cypmode.validation.pocket_geometry's module
docstring for the same kind of tradeoff made there).

BiSyRMSD's real, published definition (confirmed via the CASP16
pharma-ligand assessment paper): the binding site is every protein
residue with a heavy atom within 4 Angstrom of any reference ligand
atom; the predicted structure is superposed onto the reference by a
Kabsch (least-squares rigid) fit over those binding-site residues' own
C-alpha atoms; RMSD is then computed on the ligand atoms only, under
that fixed superposition, minimized over symmetry-equivalent atom-index
permutations (not re-aligned again -- alignment already happened via
the pocket fit).

This module reproduces exactly that recipe with tools already in this
project's own dependency stack: Bio.PDB.Superimposer for the pocket
Kabsch fit (the same real Kabsch algorithm OpenStructure itself uses,
not an approximation of it), and rdkit.Chem.rdMolAlign.CalcRMS for the
final ligand RMSD -- CalcRMS specifically (not GetBestRMS, which
cypmode.validation.pose_comparison.ligand_pose_rmsd uses for a
different purpose): CalcRMS computes a symmetry-aware RMSD "in place,"
without re-aligning the two molecules first, which is exactly what's
needed once the pocket-based superposition has already fixed the
relative frame -- RDKit's own docstring for CalcRMS names "the RMSD
between docking poses and the co-crystallized ligand" as its intended
use, the same real situation as this module's.

Real caveat (matching pose_comparison.py's own): this project doesn't
yet have two independent real structures of the same compound (one
Boltz-2 run per compound) or a real deposited reference structure
locally to compare against -- this module is real, tested
infrastructure for when either becomes available, not yet applied to
a real prediction-vs-reference comparison.
"""

import numpy as np
from Bio.PDB import Superimposer
from rdkit.Chem import rdMolAlign

from cypmode.validation.structures import (
    _STANDARD_RESIDUES,
    _ligand_atoms,
    _ligand_mol_3d,
)

BINDING_SITE_CUTOFF_A = 4.0


def binding_site_ca_atoms(structure, ligand_cutoff: float = BINDING_SITE_CUTOFF_A) -> dict:
    """Maps each protein residue with a heavy atom within ligand_cutoff A
    of any ligand atom to that residue's own C-alpha Bio.PDB Atom --
    OpenStructure's own real binding-site definition (see module
    docstring). Keyed by residue id (Bio.PDB's own (hetflag, seqid,
    icode) tuple) so two structures of the same protein -- same
    sequence, same numbering -- can be matched residue-for-residue.
    Residues without a resolved C-alpha (rare, but real in a predicted
    structure) are silently excluded, not treated as a binding-site
    match with no coordinate to superpose on.
    """
    ligand_coords = np.array([atom.get_coord() for atom in _ligand_atoms(structure)])
    if len(ligand_coords) == 0:
        raise ValueError("no ligand atoms found -- is the inhibitor chain present?")

    out = {}
    for residue in structure.get_residues():
        if residue.get_resname() not in _STANDARD_RESIDUES:
            continue
        if "CA" not in residue:
            continue
        heavy_coords = np.array([atom.get_coord() for atom in residue.get_atoms() if atom.element != "H"])
        if heavy_coords.size == 0:
            continue
        min_dist = np.linalg.norm(heavy_coords[:, None, :] - ligand_coords[None, :, :], axis=-1).min()
        if min_dist <= ligand_cutoff:
            out[residue.get_id()] = residue["CA"]
    return out


def bisy_rmsd(reference_structure, predicted_structure, ligand_smiles: str, ligand_cutoff: float = BINDING_SITE_CUTOFF_A) -> float:
    """The real BiSyRMSD recipe (see module docstring): binding site
    defined on reference_structure, Kabsch-superposes predicted_structure
    onto it via that pocket's C-alpha atoms, then reports the
    symmetry-aware ligand RMSD under that fixed superposition -- not
    re-aligned on the ligand itself, which would defeat the point of
    scoring absolute pocket placement rather than just ligand shape
    (that's what cypmode.validation.pose_comparison.ligand_pose_rmsd is
    for).

    Raises ValueError if fewer than 3 binding-site residues have a
    matching C-alpha in both structures (Superimposer's Kabsch fit is
    undefined/unstable below 3 points).
    """
    ref_ca = binding_site_ca_atoms(reference_structure, ligand_cutoff)
    # the pocket is DEFINED on the reference (see module docstring); the
    # predicted structure just needs to supply a C-alpha at each of those
    # same residue ids, not its own separately-computed pocket
    pred_all_ca = {residue.get_id(): residue["CA"] for residue in predicted_structure.get_residues() if "CA" in residue}

    common_ids = sorted(set(ref_ca) & set(pred_all_ca))
    if len(common_ids) < 3:
        raise ValueError(
            f"only {len(common_ids)} binding-site residues have a C-alpha in both structures "
            "(need >= 3 for a Kabsch fit) -- are these really the same protein/numbering?"
        )

    ref_atoms = [ref_ca[i] for i in common_ids]
    pred_atoms = [pred_all_ca[i] for i in common_ids]

    superimposer = Superimposer()
    superimposer.set_atoms(ref_atoms, pred_atoms)  # fixed=reference pocket, moving=predicted pocket
    rot, tran = superimposer.rotran

    pred_ligand_mol, _ = _ligand_mol_3d(predicted_structure, ligand_smiles)
    ref_ligand_mol, _ = _ligand_mol_3d(reference_structure, ligand_smiles)

    conf = pred_ligand_mol.GetConformer()
    for i in range(pred_ligand_mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        new_pos = pos @ rot + tran
        conf.SetAtomPosition(i, new_pos.tolist())

    return rdMolAlign.CalcRMS(pred_ligand_mol, ref_ligand_mol)
