# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Group labels for group-aware validation: cluster compounds so that whole chemical series stay on one
side of every split. Used by NestedPlan(groups=...)."""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from cypmode.challenge.validation_matching import tanimoto_matrix

RDLogger.DisableLog("rdApp.*")


def butina_groups(fingerprints: np.ndarray, similarity_cutoff: float = 0.6) -> np.ndarray:
    """Taylor-Butina clustering on Tanimoto distance: every compound ends up within ``1 - similarity_cutoff``
    of its cluster centroid. Returns one integer cluster id per row (singletons get their own id).
    """
    n = len(fingerprints)
    if n == 1:
        return np.zeros(1, dtype=np.int64)
    distance = 1.0 - tanimoto_matrix(fingerprints)
    clusters = Butina.ClusterData(distance[np.tril_indices(n, -1)].astype(float).tolist(), n, 1.0 - similarity_cutoff, isDistData=True)
    labels = np.full(n, -1, dtype=np.int64)
    for cluster_id, members in enumerate(clusters):
        labels[list(members)] = cluster_id
    return labels


def scaffold_groups(smiles: list[str], generic: bool = True) -> np.ndarray:
    """Bemis-Murcko scaffold id per compound (the carbon skeleton when ``generic``); acyclic compounds share one id."""
    ids: dict[str, int] = {}
    out = np.empty(len(smiles), dtype=np.int64)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic and scaffold.GetNumAtoms():
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        out[i] = ids.setdefault(Chem.MolToSmiles(scaffold), len(ids))
    return out


def morgan_bit_matrix(smiles: list[str], n_bits: int = 2048) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    return np.stack([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles]).astype(np.float32)
