"""Experimental-prior rescoring of a predicted inhibitor pose in CYP3A4.

The Structure Track's accuracy metrics are scored against sequestered
structures, so a predicted pose can't be scored locally. What can be asked
instead: how well does this pose agree with where experimentally resolved
ligands sit in this pocket? The reference set is the PDB's active-site
CYP3A4 co-crystal ligands (cypmode.validation.reference_ligands). Each
reference is superposed into the candidate's frame on the C-alpha atoms
around the heme (matched by residue number), then two scores are computed
from heavy-atom coordinates alone -- no bond orders or atom
correspondence needed:

template_score
    Similarity-weighted mean Gaussian-overlap Tanimoto between the
    candidate ligand and its TOP_K most chemically similar reference
    ligands (Morgan/Tanimoto weights). The classic "similar ligands bind
    similarly" template idea: strong when a close analog has been
    crystallized, uninformative when none has. `template_support` (the
    best chemical similarity found) says which of the two applies, so a
    caller never mistakes a weak template for evidence.
occupancy_score
    The fraction of ALL reference ligands that occupy the space around each
    candidate heavy atom, averaged over the candidate's atoms. A prior that
    still works for a novel scaffold: it penalizes a pose sitting where
    no experimental ligand ever has (the access channel, the protein
    surface), and says little about two poses that are both inside the
    populated active site.

Neither score looks at the protein beyond the pocket superposition, so
they complement PoseBusters (physical validity) rather than replace it.
How much each score actually discriminates a near-native pose from a wrong
one is measured, not assumed, in cypmode.validation.pose_rescoring_validation.
"""

from functools import lru_cache

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem

from cypmode.validation.structures import (
    _STANDARD_RESIDUES,
    _ligand_atoms,
    find_heme_iron,
)

SHAPE_SIGMA_A = 1.2
OCCUPANCY_SIGMA_A = 1.5
TOP_K = 5
MIN_POCKET_CA = 25
MAX_ALIGN_RMSD_A = 1.5
POCKET_RADIUS_A = 12.0


def gaussian_overlap_tanimoto(coords_a: np.ndarray, coords_b: np.ndarray, sigma: float = SHAPE_SIGMA_A) -> float:
    """First-order Gaussian volume-overlap Tanimoto between two heavy-atom
    point sets already in the same frame: every atom is a spherical
    Gaussian, V_XY = sum_ij exp(-d_ij^2 / (4 sigma^2)) (the pairwise
    product-Gaussian overlap; the constant prefactor is shared by all three
    terms and cancels), T = V_AB / (V_AA + V_BB - V_AB). 1.0 for identical
    sets, falling toward 0 as they separate or differ in size.
    """

    def volume(x, y):
        d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)
        return float(np.exp(-d2 / (4.0 * sigma**2)).sum())

    v_ab = volume(coords_a, coords_b)
    return v_ab / (volume(coords_a, coords_a) + volume(coords_b, coords_b) - v_ab)


@lru_cache(maxsize=4096)
def morgan_fingerprint(smiles: str):
    """Morgan (radius 2, 2048-bit) fingerprint, cached: the same ~100
    reference SMILES are fingerprinted for every pose scored. RDKit's log is
    muted while parsing -- some reference ligands are organometallic
    complexes whose unusual bond types make it very noisy, and None (not an
    exception) is returned for anything that doesn't parse.
    """
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol is not None else None


@lru_cache(maxsize=4096)
def connectivity_key(smiles: str) -> str | None:
    """InChIKey connectivity block (first 14 characters): the same compound
    regardless of stereochemistry/protonation notation -- how a reference
    entry for the compound under test is recognized and excluded when
    validating against its own crystal structure.
    """
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return None
        key = Chem.MolToInchiKey(mol)
    return key[:14] if key else None


def superpose_reference(reference: dict, candidate_fe: np.ndarray, candidate_ca: dict[int, np.ndarray]) -> tuple[np.ndarray, float] | None:
    """Reference ligand coordinates moved into the candidate's frame by
    superposing the reference's pocket C-alpha atoms onto the candidate's
    (matched by residue number, restricted to those within POCKET_RADIUS_A
    of the candidate's iron). Returns (coords_in_candidate_frame, ca_rmsd),
    or None if too few residues match or the pockets don't superpose
    within MAX_ALIGN_RMSD_A -- a reference that doesn't fit is dropped,
    never forced.
    """
    ref_ca = {int(k): np.asarray(v, dtype=float) for k, v in reference["pocket_ca"].items()}
    shared = sorted(
        r for r in ref_ca.keys() & candidate_ca.keys() if np.linalg.norm(candidate_ca[r] - candidate_fe) <= POCKET_RADIUS_A
    )
    if len(shared) < MIN_POCKET_CA:
        return None
    superimposer = SVDSuperimposer()
    superimposer.set(np.array([candidate_ca[r] for r in shared]), np.array([ref_ca[r] for r in shared]))
    superimposer.run()
    rms = float(superimposer.get_rms())
    if rms > MAX_ALIGN_RMSD_A:
        return None
    rotation, translation = superimposer.get_rotran()
    return np.asarray(reference["coords"], dtype=float) @ rotation + translation, rms


def align_references(
    ligand_smiles: str,
    fe: np.ndarray,
    pocket_ca: dict[int, np.ndarray],
    references: list[dict],
    exclude_compound: str | None = None,
    exclude_similarity_at_least: float | None = None,
) -> list[tuple[float | None, str, np.ndarray]]:
    """Every usable reference as (chemical similarity to the query ligand,
    "pdb:comp" label, heavy-atom coordinates in the candidate's frame).
    The superposition depends only on the pocket, not on the ligand pose,
    so this is computed once per frame and reused for every pose scored in
    it (score_against_aligned).

    `exclude_compound` (a SMILES) drops every reference that is the same
    compound, and `exclude_similarity_at_least` additionally drops
    references at least that chemically similar to the query -- for
    validating against a ligand's own crystal structure, optionally
    mimicking a novel scaffold with no analogs.
    """
    own_key = connectivity_key(exclude_compound) if exclude_compound else None
    query_fp = morgan_fingerprint(ligand_smiles)

    aligned = []
    for reference in references:
        if own_key is not None and connectivity_key(reference["smiles"]) == own_key:
            continue
        superposed = superpose_reference(reference, fe, pocket_ca)
        if superposed is None:
            continue
        ref_fp = morgan_fingerprint(reference["smiles"]) if reference.get("smiles") else None
        similarity = DataStructs.TanimotoSimilarity(query_fp, ref_fp) if query_fp is not None and ref_fp is not None else None
        if exclude_similarity_at_least is not None and similarity is not None and similarity >= exclude_similarity_at_least:
            continue
        aligned.append((similarity, f"{reference['pdb_id']}:{reference['comp_id']}", superposed[0]))
    return aligned


def score_against_aligned(ligand_coords: np.ndarray, aligned: list, top_k: int = TOP_K) -> dict:
    """template_score / occupancy_score (see module docstring) for one pose
    against references already in its frame. Returns template_score (None
    if no reference has a usable fingerprint), template_support (best
    chemical similarity among the references used), top_template,
    occupancy_score and n_refs_used.
    """
    result = {"template_score": None, "template_support": None, "top_template": None, "occupancy_score": None, "n_refs_used": len(aligned)}
    if not aligned:
        return result

    scored = sorted((a for a in aligned if a[0] is not None), key=lambda a: -a[0])[:top_k]
    if scored and sum(s for s, _, _ in scored) > 0:
        overlaps = [gaussian_overlap_tanimoto(ligand_coords, coords) for _, _, coords in scored]
        weights = [s for s, _, _ in scored]
        result["template_score"] = float(np.average(overlaps, weights=weights))
        result["template_support"] = float(scored[0][0])
        result["top_template"] = scored[0][1]

    occupancy = []
    for _, _, coords in aligned:
        d2 = ((ligand_coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1)
        occupancy.append(np.exp(-d2 / (2.0 * OCCUPANCY_SIGMA_A**2)).max(axis=1))  # per candidate atom: closest reference atom's kernel
    result["occupancy_score"] = float(np.mean(occupancy))
    return result


def score_pose_arrays(
    ligand_coords: np.ndarray,
    fe: np.ndarray,
    pocket_ca: dict[int, np.ndarray],
    ligand_smiles: str,
    references: list[dict],
    exclude_compound: str | None = None,
    exclude_similarity_at_least: float | None = None,
    top_k: int = TOP_K,
) -> dict:
    """Both scores for one ligand pose given as plain arrays (heavy-atom
    coordinates, heme iron, {residue number: C-alpha coordinate}).
    """
    aligned = align_references(ligand_smiles, fe, pocket_ca, references, exclude_compound, exclude_similarity_at_least)
    return score_against_aligned(ligand_coords, aligned, top_k)


def structure_arrays(structure) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """(ligand heavy-atom coordinates, heme iron, {residue number: C-alpha}) from a
    predicted complex -- the same ligand/heme/protein conventions as
    cypmode.validation.structures. Residue numbers are those of the first
    protein chain (the prediction's chain A, UniProt numbering).
    """
    ligand = np.array([a.coord for a in _ligand_atoms(structure) if a.element.strip().upper() not in ("H", "D")], dtype=float)
    fe = np.asarray(find_heme_iron(structure).coord, dtype=float)
    chain = next(c for c in structure[0] if any(r.resname in _STANDARD_RESIDUES for r in c))
    pocket_ca = {r.id[1]: np.asarray(r["CA"].coord, dtype=float) for r in chain if r.resname in _STANDARD_RESIDUES and "CA" in r}
    return ligand, fe, pocket_ca


def rescore_pose(structure, ligand_smiles: str, references: list[dict], **kwargs) -> dict:
    """score_pose_arrays on a predicted complex (Bio.PDB structure)."""
    ligand, fe, pocket_ca = structure_arrays(structure)
    return score_pose_arrays(ligand, fe, pocket_ca, ligand_smiles, references, **kwargs)
