# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Analog series for structure prediction: SAR-aware pose consistency (jobs 133, 142, 169).

Compounds of one chemical series bind the same pocket in related ways: the shared core (maximum common substructure) sits in nearly the
same place. A predicted pose of one analog is therefore evidence about the poses of the others. This module provides what the consensus
rule needs: the common core of two ligands, the core RMSD between a pose of one and a pose of the other after the two predicted complexes
are superposed on their protein C-alpha atoms, and the selection of crystal-anchored analog pairs for a benchmark.

Benchmark pairs (``select_series_benchmark``) come from the experimental CYP3A4 reference set: two different compounds (different
connectivity), Tanimoto 0.5-0.9, 20-40 heavy atoms, neither in the existing benchmark, no ligand used twice, at most two pairs from one
name family; both members have crystal structures, so every predicted pose has a true RMSD.
"""

import re

import numpy as np
import pandas as pd

TANIMOTO_RANGE = (0.5, 0.9)
HEAVY_RANGE = (20, 40)


def select_series_benchmark(references: list[dict], exclude_smiles: set[str], n_pairs: int = 3) -> pd.DataFrame:
    """Columns: Molecule_Name (``series-<component id>``), SMILES, series (pair number)."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    from cypmode.validation.pose_rescoring import connectivity_key

    keys_out = {connectivity_key(s) for s in exclude_smiles}
    distinct = {}
    for r in references:
        key = connectivity_key(r["smiles"])
        if key and key not in keys_out and HEAVY_RANGE[0] <= r["n_heavy"] <= HEAVY_RANGE[1]:
            distinct.setdefault(key, r)
    items = list(distinct.values())
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(r["smiles"]), 2, 2048) for r in items]
    candidates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            t = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if TANIMOTO_RANGE[0] <= t <= TANIMOTO_RANGE[1]:
                candidates.append((abs(t - 0.7), i, j))
    used, families, rows = set(), {}, []
    for _, i, j in sorted(candidates):
        family = re.sub(r"[^a-z]", "", items[i]["name"].lower())[:8]
        if i in used or j in used or families.get(family, 0) >= 2 or len(rows) >= 2 * n_pairs:
            continue
        used |= {i, j}
        families[family] = families.get(family, 0) + 1
        for k in (i, j):
            rows.append({"Molecule_Name": f"series-{items[k]['comp_id']}", "SMILES": items[k]["smiles"], "series": len(rows) // 2 + 1})
    return pd.DataFrame(rows, columns=["Molecule_Name", "SMILES", "series"])


def common_core(smiles_a: str, smiles_b: str, timeout: int = 10) -> list[tuple[int, int]]:
    """Atom index pairs (in ``smiles_a`` and ``smiles_b`` order) of the maximum common substructure (rings match rings, whole rings only)."""
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    a, b = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    result = rdFMCS.FindMCS([a, b], timeout=timeout, ringMatchesRingOnly=True, completeRingsOnly=True, matchValences=False)
    if result.numAtoms < 5:
        return []
    pattern = Chem.MolFromSmarts(result.smartsString)
    return list(zip(a.GetSubstructMatch(pattern), b.GetSubstructMatch(pattern), strict=False))


def core_rmsd(structure_a, smiles_a: str, structure_b, smiles_b: str, core: list[tuple[int, int]], pocket_radius: float = 12.0) -> float:
    """RMSD over the common core between a pose of ligand A and a pose of ligand B, the two complexes superposed on pocket C-alpha atoms."""
    from rdkit import Chem

    from cypmode.validation.pose_rescoring import structure_arrays
    from cypmode.validation.structures import _ligand_mol_3d

    _, fe_a, ca_a = structure_arrays(structure_a)
    _, _fe_b, ca_b = structure_arrays(structure_b)
    shared = sorted(set(ca_a) & set(ca_b))
    near = [r for r in shared if np.linalg.norm(ca_a[r] - fe_a) <= pocket_radius]
    if len(near) < 8 or not core:
        return float("nan")
    pa, pb = np.array([ca_a[r] for r in near]), np.array([ca_b[r] for r in near])
    mol_a, _ = _ligand_mol_3d(structure_a, smiles_a)
    mol_b, _ = _ligand_mol_3d(structure_b, smiles_b)
    match_a = mol_a.GetSubstructMatch(Chem.MolFromSmiles(smiles_a))
    match_b = mol_b.GetSubstructMatch(Chem.MolFromSmiles(smiles_b))
    if not match_a or not match_b:
        return float("nan")
    xa, xb = mol_a.GetConformer().GetPositions(), mol_b.GetConformer().GetPositions()
    xb = (xb - pb.mean(axis=0)) @ _rotation(pb, pa) + pa.mean(axis=0)  # complex B superposed on complex A
    ia, ib = [match_a[i] for i, _ in core], [match_b[j] for _, j in core]
    return float(np.sqrt(((xa[ia] - xb[ib]) ** 2).sum(axis=1).mean()))


def _rotation(moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that (moving - centroid) @ R best matches (fixed - centroid)."""
    u, _, vt = np.linalg.svd((moving - moving.mean(axis=0)).T @ (fixed - fixed.mean(axis=0)))
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Write the analog-pair benchmark (structure_benchmark_series.csv) from the crystal reference set.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--out", default="data/structure_track/inputs/structure_benchmark_series.csv")
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if args.build:
        references = json.loads(Path("data/structure_track/inputs/cyp3a4_reference_ligands.json").read_text())["ligands"]
        existing = set(pd.read_csv("data/structure_track/inputs/structure_benchmark.csv").SMILES)
        table = select_series_benchmark(references, existing, args.pairs)
        table.to_csv(args.out, index=False)
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
