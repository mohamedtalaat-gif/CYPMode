# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Pose pools of one ligand and the pairwise LDDT-PLI matrix inside them (shared by the structure-track gates).

``pairwise_lddt`` returns M with M[i, j] = approximate LDDT-PLI of pose i when pose j is taken as the truth
(``pose_selection.lddt_pli_arrays``; not symmetric because the reference defines which protein atoms are included).
The matrix is cached on disk keyed by the path, size and modification time of every file, so the gates that reuse the same
pools (selection validation, heme prior, expected-improvement sampling) pay for it once.

Poses of one engine share their ligand atom names. Poses of two engines do not (Boltz-2 names the inhibitor atoms C42, C48, ...,
OpenFold3 names them C1, C2, ...), so a pool that mixes engines is compared through the ligand's SMILES: every pose is rebuilt on the
SMILES template, its atoms are labelled by template index, and the LDDT-PLI of a pair is the best value over the template's
symmetry-equivalent atom mappings (a flipped phenyl or a rotated tert-butyl is not an error). Pass ``smiles`` to ``pairwise_lddt``
for such pools; pools of one engine keep the name-based computation and their cached matrices.
"""

import hashlib
from pathlib import Path

import numpy as np

POOL_CACHE = Path("data/structure_track/outputs/cyp3a4/pose_pool_cache")


def pool_paths(name: str, engines: tuple[str, ...] = ("boltz",), target: str = "cyp3a4") -> list[tuple[str, int, Path]]:
    """[(engine, sample index, CIF path)] for one compound, in sample order, restricted to ``engines``."""
    from cypmode.validation.openfold3 import model_cif_paths
    from cypmode.validation.pose_consensus import boltz_cif_paths

    out: list[tuple[str, int, Path]] = []
    if "boltz" in engines:
        out += [("boltz", i, p) for i, p in enumerate(boltz_cif_paths(name, target))]
    if "openfold3" in engines:
        out += [("openfold3", i, p) for i, p in enumerate(model_cif_paths(name, target))]
    return out


def _key(paths: list[Path]) -> str:
    digest = hashlib.md5()
    for p in paths:
        stat = Path(p).stat()
        digest.update(f"{p}|{stat.st_size}|{int(stat.st_mtime)}".encode())
    return digest.hexdigest()


MAX_SYMMETRY_MAPPINGS = 512  # automorphisms of the ligand graph considered per pair (drug-like ligands have far fewer)


def template_automorphisms(smiles: str, limit: int = MAX_SYMMETRY_MAPPINGS) -> np.ndarray:
    """Atom permutations of the SMILES graph that map the molecule onto itself, shape (M, K); row 0 is the identity."""
    from rdkit import Chem

    template = Chem.MolFromSmiles(smiles)
    matches = template.GetSubstructMatches(template, uniquify=False, useChirality=False, maxMatches=limit)
    perms = np.array(matches, dtype=int)
    identity = np.arange(template.GetNumAtoms())
    order = np.argsort(~(perms == identity).all(axis=1), kind="stable")  # identity first
    return perms[order]


def template_ligand_coordinates(structure, smiles: str) -> np.ndarray:
    """Heavy-atom coordinates of the predicted inhibitor ordered by the atoms of the SMILES template, shape (K, 3)."""
    from rdkit import Chem

    from cypmode.validation.structures import _ligand_mol_3d

    mol, _ = _ligand_mol_3d(structure, smiles)
    template = Chem.MolFromSmiles(smiles)
    match = mol.GetSubstructMatch(template)
    if len(match) != template.GetNumAtoms():
        raise ValueError("the predicted ligand does not match its SMILES template")
    conformer = mol.GetConformer()
    return np.array([list(conformer.GetAtomPosition(i)) for i in match], dtype=float)


def symmetric_lddt_pli(model_ligand: np.ndarray, model_protein: dict, reference_ligand: np.ndarray, reference_protein: dict, permutations: np.ndarray) -> float:
    """LDDT-PLI of a model against a reference, best over the symmetry mappings of the ligand (arrays ordered by template atom)."""
    from cypmode.validation.pose_selection import INCLUSION_RADIUS_A, THRESHOLDS_A

    keys = sorted(set(model_protein) & set(reference_protein))
    if not keys:
        raise ValueError("model and reference share no protein atoms")
    model_xyz = np.array([model_protein[k] for k in keys])
    reference_xyz = np.array([reference_protein[k] for k in keys])
    reference_d = np.linalg.norm(reference_ligand[:, None, :] - reference_xyz[None, :, :], axis=2)
    included = reference_d < INCLUSION_RADIUS_A
    if not included.any():
        return 0.0
    columns = included.any(axis=0)  # only protein atoms near the reference ligand can enter the score
    model_d = np.linalg.norm(model_ligand[:, None, :] - model_xyz[None, columns, :], axis=2)
    reference_d, included = reference_d[:, columns], included[:, columns]
    best = 0.0
    for perm in permutations:
        error = np.abs(model_d[perm] - reference_d)[included]
        best = max(best, float(np.mean([np.mean(error < t) for t in THRESHOLDS_A])))
    return best


def pairwise_lddt(paths: list[Path], cache_dir: Path = POOL_CACHE, smiles: str | None = None) -> np.ndarray:
    from cypmode.validation.pose_selection import complex_arrays, lddt_pli_arrays
    from cypmode.validation.structures import load_structure

    cache_dir.mkdir(parents=True, exist_ok=True)
    structures = None
    by_template = False
    if smiles is not None:
        structures = [load_structure(p) for p in paths]
        names = [frozenset(complex_arrays(s)[0]) for s in structures]
        by_template = len(set(names)) > 1  # ligand atom names differ between poses: a pool that mixes engines
    file = cache_dir / f"{_key(paths)}{'_template' if by_template else ''}.npy"
    if file.exists():
        return np.load(file)
    if by_template:
        permutations = template_automorphisms(smiles)
        ligands = [template_ligand_coordinates(s, smiles) for s in structures]
        proteins = [complex_arrays(s)[1] for s in structures]
        n = len(paths)
        matrix = np.eye(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i, j] = symmetric_lddt_pli(ligands[i], proteins[i], ligands[j], proteins[j], permutations)
        np.save(file, matrix)
        return matrix
    complexes = [complex_arrays(s) for s in (structures if structures is not None else [load_structure(p) for p in paths])]
    n = len(complexes)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = lddt_pli_arrays(*complexes[i], *complexes[j])
    np.save(file, matrix)
    return matrix


def mbr_pick(matrix: np.ndarray, subset: list[int] | None = None) -> int:
    """Index (into the full matrix) of the pose with the highest mean LDDT-PLI against the other members of ``subset``."""
    members = list(range(len(matrix))) if subset is None else list(subset)
    if len(members) < 2:
        return members[0]
    scores = [float(np.mean([matrix[i, j] for j in members if j != i])) for i in members]
    return members[int(np.argmax(scores))]


def pool_disagreement(matrix: np.ndarray, subset: list[int]) -> float:
    """1 - the best expected LDDT-PLI in the subset: how far the pool is from agreeing on one pose (1 sample -> 1.0)."""
    if len(subset) < 2:
        return 1.0
    return float(1.0 - max(np.mean([matrix[i, j] for j in subset if j != i]) for i in subset))
