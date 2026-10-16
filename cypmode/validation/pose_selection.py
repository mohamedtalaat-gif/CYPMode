"""Choose one pose from a pool by the metric the challenge scores, instead of the model's own confidence.

The structure track is scored by LDDT-PLI: for the protein atoms within 6 A of the ligand in the true
complex, how well are their distances to the ligand atoms preserved (thresholds 0.5, 1, 2 and 4 A).
Boltz-2's confidence is 0.91-0.96 even on poses 5-6 A from the crystal, so it cannot rank a pool.
Minimum-Bayes-risk selection treats every pose in the pool as a plausible truth and submits the pose with
the highest expected LDDT-PLI against all the others: it is the pool's consensus, and it needs no
confidence score at all.

This LDDT-PLI is an approximation of the official one (no OpenStructure): ligand atoms are matched by
atom name, so symmetry-equivalent atoms are not remapped, and only heavy atoms count. It is used to
compare predicted poses with each other, never to report a score.
"""

from pathlib import Path

import numpy as np

THRESHOLDS_A = (0.5, 1.0, 2.0, 4.0)
INCLUSION_RADIUS_A = 6.0


def lddt_pli_arrays(
    model_ligand: dict[str, np.ndarray],
    model_protein: dict[tuple, np.ndarray],
    reference_ligand: dict[str, np.ndarray],
    reference_protein: dict[tuple, np.ndarray],
) -> float:
    """Approximate LDDT-PLI of a model complex against a reference complex (atom name -> coordinates)."""
    ligand_names = sorted(set(model_ligand) & set(reference_ligand))
    protein_keys = sorted(set(model_protein) & set(reference_protein))
    if not ligand_names or not protein_keys:
        raise ValueError("model and reference share no ligand or protein atoms")

    def distances(ligand, protein):
        lig = np.array([ligand[n] for n in ligand_names])
        prot = np.array([protein[k] for k in protein_keys])
        return np.linalg.norm(lig[:, None, :] - prot[None, :, :], axis=2)

    reference_d = distances(reference_ligand, reference_protein)
    model_d = distances(model_ligand, model_protein)
    included = reference_d < INCLUSION_RADIUS_A
    if not included.any():
        return 0.0
    error = np.abs(model_d - reference_d)[included]
    return float(np.mean([np.mean(error < t) for t in THRESHOLDS_A]))


def complex_arrays(structure) -> tuple[dict[str, np.ndarray], dict[tuple, np.ndarray]]:
    """(ligand heavy atoms by name, protein heavy atoms by (residue number, name)) of a predicted complex."""
    from cypmode.validation.structures import _STANDARD_RESIDUES, _ligand_atoms

    ligand = {a.get_name(): np.asarray(a.coord, dtype=float) for a in _ligand_atoms(structure) if a.element.strip().upper() not in ("H", "D")}
    chain = next(c for c in structure[0] if any(r.resname in _STANDARD_RESIDUES for r in c))
    protein = {
        (r.id[1], a.get_name()): np.asarray(a.coord, dtype=float)
        for r in chain
        if r.resname in _STANDARD_RESIDUES
        for a in r
        if a.element.strip().upper() not in ("H", "D")
    }
    return ligand, protein


def expected_lddt(complexes: list[tuple[dict, dict]], weights: np.ndarray | None = None) -> np.ndarray:
    """Expected LDDT-PLI of each pose when every other pose in the pool is taken as the truth."""
    n = len(complexes)
    if n < 2:
        return np.ones(n)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    scores = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        values = np.array([lddt_pli_arrays(*complexes[i], *complexes[j]) for j in others])
        scores[i] = float(np.average(values, weights=w[others]))
    return scores


def select_pose(cif_paths: list[Path], weights: np.ndarray | None = None) -> tuple[int, np.ndarray]:
    """Index of the pose with the highest expected LDDT-PLI in the pool, and every pose's expected score."""
    from cypmode.validation.structures import load_structure

    scores = expected_lddt([complex_arrays(load_structure(p)) for p in cif_paths], weights)
    return int(scores.argmax()), scores
