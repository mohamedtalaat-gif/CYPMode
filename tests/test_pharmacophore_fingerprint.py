# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pmapper")

from cypmode.challenge.pharmacophore_fingerprint import (
    PMAPPER_NBITS,
    _embed_3d_conformer,
    add_pharmacophore_fingerprints,
    compute_pharmacophore_fingerprint,
    compute_pharmacophore_fingerprints_batch,
)

KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"


def test_embed_3d_conformer_returns_a_real_3d_mol():
    mol = _embed_3d_conformer(KETOCONAZOLE)
    assert mol is not None
    assert mol.GetNumConformers() == 1


def test_embed_3d_conformer_returns_none_for_unparseable_smiles():
    assert _embed_3d_conformer("not a real smiles !!!") is None


def test_compute_pharmacophore_fingerprint_is_a_real_sparse_binary_vector():
    fp = compute_pharmacophore_fingerprint(KETOCONAZOLE)
    assert fp.shape == (PMAPPER_NBITS,)
    assert set(np.unique(fp)) <= {0.0, 1.0}
    # a real, complex drug molecule should activate a real number of bits,
    # not be all zero (that would mean the pipeline silently failed) and
    # not be all one (that would mean the hashing is broken)
    assert 0 < fp.sum() < PMAPPER_NBITS


def test_compute_pharmacophore_fingerprint_returns_all_zero_for_unembeddable_smiles():
    fp = compute_pharmacophore_fingerprint("not a real smiles !!!")
    assert fp.shape == (PMAPPER_NBITS,)
    assert fp.sum() == 0


def test_compute_pharmacophore_fingerprint_distinguishes_two_different_real_molecules():
    ethanol_fp = compute_pharmacophore_fingerprint(ETHANOL)
    benzene_fp = compute_pharmacophore_fingerprint(BENZENE)
    assert not np.array_equal(ethanol_fp, benzene_fp)


def test_compute_pharmacophore_fingerprints_batch_caches_to_disk_and_resumes(tmp_path):
    cache_path = tmp_path / "cache.csv"
    first = compute_pharmacophore_fingerprints_batch([ETHANOL, BENZENE], cache_path=cache_path)
    assert len(first) == 2
    assert cache_path.exists()
    assert len(pd.read_csv(cache_path)) == 2

    second = compute_pharmacophore_fingerprints_batch([ETHANOL, "C"], cache_path=cache_path)
    assert len(second) == 2
    assert len(pd.read_csv(cache_path)) == 3  # ethanol, benzene, methane


def test_compute_pharmacophore_fingerprints_batch_works_with_multiple_workers(tmp_path):
    # regression test: the real n_workers=8 batch run crashed with
    # AttributeError because the compute callable passed to
    # multiprocessing.Pool.imap was a local lambda, which is not
    # picklable -- n_workers=1 (every other test here) never exercises
    # that code path, so this must use n_workers > 1 to actually catch it.
    cache_path = tmp_path / "cache.csv"
    result = compute_pharmacophore_fingerprints_batch([ETHANOL, BENZENE, "C"], cache_path=cache_path, n_workers=2)
    assert len(result) == 3
    assert len(pd.read_csv(cache_path)) == 3


def test_compute_pharmacophore_fingerprints_batch_checkpoints_mid_run(tmp_path):
    cache_path = tmp_path / "cache.csv"
    compute_pharmacophore_fingerprints_batch([ETHANOL, BENZENE, "C"], cache_path=cache_path, checkpoint_every=1)
    assert len(pd.read_csv(cache_path)) == 3


def test_add_pharmacophore_fingerprints_adds_one_fixed_size_array_per_row(tmp_path):
    df = pd.DataFrame({"SMILES": [ETHANOL, BENZENE]})
    out = add_pharmacophore_fingerprints(df, cache_path=tmp_path / "cache.csv")
    assert "pharmacophore_fp" in out.columns
    assert all(len(fp) == PMAPPER_NBITS for fp in out["pharmacophore_fp"])
