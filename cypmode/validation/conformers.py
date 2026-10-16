"""Distinct crystal conformations of the CYP3A4 pocket, to condition co-folding on more than one of them.

Unguided Boltz-2 returns five near-identical poses per ligand (on the diverse crystal benchmark the best
of five is only ~0.7 A better than the first pick), so no re-ranking can rescue a wrong one. CYP3A4's
pocket changes shape between crystal structures; conditioning the prediction on a different crystal
conformation is a way to make the pool differ where it matters. This module picks those conformations.

A benchmark ligand comes from the PDB, so its own entries -- and entries whose ligand is a close analog --
are never offered as templates: the template must not carry the answer.
"""

from pathlib import Path

import numpy as np
from rdkit import DataStructs

from cypmode.validation.pose_rescoring import connectivity_key, morgan_fingerprint
from cypmode.validation.reference_ligands import CACHE_DIR

MIN_SHARED_RESIDUES = 20  # pocket residues two entries must both resolve for their distance to mean anything
MAX_RESOLUTION_A = 3.2
CRYOEM_DIR = Path("data/structure_track/inputs/cryoem_templates")
CRYOEM_TEMPLATE_CHAIN = "A"


def _kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((p @ rotation - q) ** 2, axis=1))))


def entry_pockets(references: list[dict], max_resolution: float = MAX_RESOLUTION_A) -> dict[str, dict[int, np.ndarray]]:
    """{pdb_id: {residue number: C-alpha}} for the pocket residues of each entry (one record per entry)."""
    pockets = {}
    for record in references:
        if record["pdb_id"] in pockets or (record.get("resolution") or 99) > max_resolution:
            continue
        pockets[record["pdb_id"]] = {int(k): np.asarray(v, dtype=float) for k, v in record["pocket_ca"].items()}
    return pockets


def pocket_distance(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> float | None:
    """C-alpha RMSD (A) of two entries' pockets over the residues both resolve, after superposition."""
    shared = sorted(set(a) & set(b))
    if len(shared) < MIN_SHARED_RESIDUES:
        return None
    return _kabsch_rmsd(np.array([a[r] for r in shared]), np.array([b[r] for r in shared]))


def excluded_entries(smiles: str, references: list[dict], similarity_at_least: float = 0.5) -> set[str]:
    """Entries whose ligand is this compound or a close analog of it -- unusable as templates for it."""
    key = connectivity_key(smiles)
    fingerprint = morgan_fingerprint(smiles)
    excluded = set()
    for record in references:
        if not record.get("smiles"):
            continue
        same = key is not None and connectivity_key(record["smiles"]) == key
        close = DataStructs.TanimotoSimilarity(fingerprint, morgan_fingerprint(record["smiles"])) >= similarity_at_least
        if same or close:
            excluded.add(record["pdb_id"])
    return excluded


def pick_conformers(pockets: dict[str, dict[int, np.ndarray]], n: int, exclude: set[str] | frozenset[str] = frozenset(), cache_dir: Path = CACHE_DIR) -> list[str]:
    """``n`` entries that differ most in pocket shape: the medoid first, then farthest-first. Only entries
    with a cached structure file and not in ``exclude`` are eligible.
    """
    eligible = sorted(e for e in pockets if e not in exclude and (Path(cache_dir) / f"{e}.cif").exists())
    if len(eligible) <= n:
        return eligible
    distance = np.full((len(eligible), len(eligible)), np.nan)
    for i, a in enumerate(eligible):
        for j in range(i + 1, len(eligible)):
            distance[i, j] = distance[j, i] = pocket_distance(pockets[a], pockets[eligible[j]]) or np.nan
    np.fill_diagonal(distance, 0.0)
    distance = np.where(np.isnan(distance), np.nanmax(distance), distance)  # unresolved pairs count as far apart
    chosen = [int(distance.mean(axis=1).argmin())]
    while len(chosen) < n:
        chosen.append(int(distance[:, chosen].min(axis=1).argmax()))
    return [eligible[i] for i in chosen]


def template_spec(pdb_id: str, cache_dir: Path = CACHE_DIR, cryoem_dir: Path = CRYOEM_DIR) -> tuple[Path, str | None]:
    """File of a template entry and the chain of that file to use (None: the file holds one protein chain).

    The cryo-EM entries are trimers; chain A is the one used, like a single crystal monomer. The cryo-EM directory is searched first so a
    structure that also sits in the crystal cache is still used as a one-chain template.
    """
    path = (Path(cryoem_dir) / f"{pdb_id}.cif").resolve()
    if path.exists():
        return path, CRYOEM_TEMPLATE_CHAIN
    path = (Path(cache_dir) / f"{pdb_id}.cif").resolve()
    if not path.exists():
        raise FileNotFoundError(f"no cached structure for {pdb_id} at {path}")
    return path, None


def template_path(pdb_id: str, cache_dir: Path = CACHE_DIR, cryoem_dir: Path = CRYOEM_DIR) -> Path:
    return template_spec(pdb_id, cache_dir, cryoem_dir)[0]
