# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pytest
from rdkit import Chem

from cypmode.challenge import frozen_embeddings as fe


def _cache(tmp_path, smiles, dim=12):
    keys = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles]
    path = tmp_path / "chem.npz"
    np.savez_compressed(path, smiles=np.array(keys, dtype=object), embeddings=np.random.default_rng(0).normal(size=(len(keys), dim)).astype(np.float32))
    return path


SMILES = ["CCO", "c1ccccc1", "CCN", "CC(=O)O", "c1ccncc1", "CCCC", "COC", "CCS"]


def test_pca_block_has_requested_width_follows_the_requested_order_and_canonicalises(tmp_path):
    path = _cache(tmp_path, SMILES)
    out = fe.load_chemeleon_pca(["OCC", "c1ccccc1"], path, n_components=3)  # "OCC" is another spelling of "CCO"
    full = fe.load_chemeleon_pca(SMILES, path, n_components=3)
    assert out.shape == (2, 3) and np.allclose(out[0], full[0]) and np.allclose(out[1], full[1])


def test_missing_cache_or_compound_is_an_error_not_a_silent_nan(tmp_path):
    with pytest.raises(FileNotFoundError, match="--extract"):
        fe.load_chemeleon_pca(["CCO"], tmp_path / "none.npz")
    with pytest.raises(ValueError, match="no CheMeleon embedding"):
        fe.load_chemeleon_pca(["CCCCCC"], _cache(tmp_path, SMILES), n_components=3)


def test_the_reduction_is_label_free_and_deterministic(tmp_path):
    path = _cache(tmp_path, SMILES)
    a = fe.load_chemeleon_pca(SMILES, path, n_components=3)
    b = fe.load_chemeleon_pca(SMILES, path, n_components=3)
    assert np.array_equal(a, b)
