# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The Atom-JEPA block reads a cache without importing torch: label-free PCA of all cached compounds, rows in the order asked for, a missing embedding replaced by the mean."""

import numpy as np
import pytest

from cypmode.challenge.atomjepa_embeddings import load_atomjepa_pca


def write_cache(path, smiles, rng, nan_row=None):
    values = rng.normal(size=(len(smiles), 8, 256)).astype(np.float32)
    if nan_row is not None:
        values[nan_row] = np.nan
    np.savez_compressed(path, smiles=np.array(smiles, dtype=object), values=values, failed=np.array([], dtype=object))


def test_rows_follow_the_request_and_the_reduction_has_the_requested_width(tmp_path):
    rng = np.random.default_rng(0)
    smiles = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "CC(=O)O", "CCOC", "CCS"]
    write_cache(tmp_path / "c.npz", smiles, rng)
    forward = load_atomjepa_pca(smiles, tmp_path / "c.npz", n_components=4)
    backward = load_atomjepa_pca(smiles[::-1], tmp_path / "c.npz", n_components=4)
    assert forward.shape == (8, 4) and np.allclose(forward, backward[::-1])
    assert np.allclose(forward.mean(axis=0), 0, atol=1e-6)  # the reduction is centred on the cached compounds


def test_a_compound_without_an_embedding_gets_the_mean_and_an_unknown_compound_is_refused(tmp_path):
    rng = np.random.default_rng(1)
    smiles = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "CC(=O)O"]
    write_cache(tmp_path / "c.npz", smiles, rng, nan_row=2)
    reduced = load_atomjepa_pca(smiles, tmp_path / "c.npz", n_components=3)
    assert np.isfinite(reduced).all()
    with pytest.raises(ValueError, match="no Atom-JEPA embedding"):
        load_atomjepa_pca(["CCBr"], tmp_path / "c.npz", n_components=3)
    with pytest.raises(FileNotFoundError):
        load_atomjepa_pca(["CCO"], tmp_path / "missing.npz")
