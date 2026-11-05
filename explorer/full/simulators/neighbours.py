# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Section 7.3: reading across. The labels of the nearest neighbours of a compound in the training file, by the Tanimoto similarity of Morgan fingerprints, and what a vote of the neighbours says."""

import numpy as np
from rdkit import Chem, DataStructs

from simulators.fingerprints import fingerprint


def prepare(smiles_list) -> list:
    """The fingerprints of the training compounds (a compound that cannot be read gets an empty one)."""
    out = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        out.append(fingerprint(m) if m is not None else DataStructs.ExplicitBitVect(2048))
    return out


def nearest(smiles: str, fingerprints: list, k: int) -> list[tuple[int, float]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    sims = np.array(DataStructs.BulkTanimotoSimilarity(fingerprint(mol), fingerprints))
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


def vote(labels: list[float], similarities: list[float], weighted: bool) -> float | None:
    """The share of positive neighbours, optionally weighted by similarity (compounds without a label are left out)."""
    pairs = [(l, s) for l, s in zip(labels, similarities, strict=True) if not np.isnan(l)]
    if not pairs:
        return None
    weights = np.array([s if weighted else 1.0 for _, s in pairs])
    return float(np.dot(weights, [l for l, _ in pairs]) / weights.sum()) if weights.sum() > 0 else None
