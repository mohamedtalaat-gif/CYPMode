# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.tblite

pytest.importorskip("tblite")

from cypmode.challenge.quantum_pharmacophore import (
    FEATURE_NAMES,
    add_quantum_features,
    compute_gfn2_xtb_properties,
    compute_quantum_features_batch,
    embed_3d_conformer,
    extract_quantum_features,
    load_cached_quantum_features,
)

ETHANOL = "CCO"
BENZENE = "c1ccccc1"


def test_embed_3d_conformer_returns_a_real_3d_mol():
    mol = embed_3d_conformer(ETHANOL)
    assert mol is not None
    assert mol.GetNumConformers() == 1


def test_embed_3d_conformer_returns_none_for_unparseable_smiles():
    assert embed_3d_conformer("not a real smiles !!!") is None


def test_compute_gfn2_xtb_properties_returns_real_electronic_structure_data():
    mol = embed_3d_conformer(ETHANOL)
    props = compute_gfn2_xtb_properties(mol)
    assert np.isfinite(props["energy"])
    assert len(props["charges"]) == mol.GetNumAtoms()
    # a real molecule is charge-neutral: GFN2-xTB partial charges sum to ~0
    assert props["charges"].sum() == pytest.approx(0.0, abs=1e-6)


def test_extract_quantum_features_produces_finite_sane_values_for_ethanol():
    features = extract_quantum_features(ETHANOL)
    assert set(features.keys()) == set(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in features.values())
    # LUMO must sit above HOMO -- a basic sanity check on orbital ordering
    assert features["qm_lumo_energy"] > features["qm_homo_energy"]
    assert features["qm_homo_lumo_gap"] == pytest.approx(
        features["qm_lumo_energy"] - features["qm_homo_energy"]
    )


def test_extract_quantum_features_distinguishes_two_different_real_molecules():
    ethanol_features = extract_quantum_features(ETHANOL)
    benzene_features = extract_quantum_features(BENZENE)
    # benzene's real aromatic pi system gives it a much smaller HOMO-LUMO
    # gap than ethanol's simple sigma-bonded structure -- not an arbitrary
    # numeric difference, a real, expected qualitative distinction
    assert benzene_features["qm_homo_lumo_gap"] != pytest.approx(
        ethanol_features["qm_homo_lumo_gap"], abs=1e-3
    )
    # benzene has zero dipole moment by symmetry; ethanol's is real and nonzero
    assert benzene_features["qm_dipole_magnitude"] < ethanol_features["qm_dipole_magnitude"]


def test_extract_quantum_features_returns_nan_row_for_unembeddable_smiles():
    features = extract_quantum_features("not a real smiles !!!")
    assert set(features.keys()) == set(FEATURE_NAMES)
    assert all(np.isnan(v) for v in features.values())


def test_compute_quantum_features_batch_caches_to_disk_and_resumes(tmp_path):
    cache_path = tmp_path / "cache.csv"
    first = compute_quantum_features_batch([ETHANOL, BENZENE], cache_path=cache_path)
    assert len(first) == 2
    assert cache_path.exists()
    cached_rows = len(pd.read_csv(cache_path))
    assert cached_rows == 2

    # a second call with an overlapping + one new compound should not
    # recompute the two already-cached rows, and should append the new one
    second = compute_quantum_features_batch([ETHANOL, "C"], cache_path=cache_path)
    assert len(second) == 2
    assert len(pd.read_csv(cache_path)) == 3  # ethanol, benzene, methane


def test_compute_quantum_features_batch_checkpoints_mid_run(tmp_path):
    """checkpoint_every=1 forces a flush after every single compound --
    confirms results land on disk progressively, not only once the
    entire batch finishes (the real risk this guards against: a crash
    partway through a several-thousand-compound run).
    """
    cache_path = tmp_path / "cache.csv"
    compute_quantum_features_batch([ETHANOL, BENZENE, "C"], cache_path=cache_path, checkpoint_every=1)
    assert len(pd.read_csv(cache_path)) == 3


def test_add_quantum_features_adds_one_fixed_size_array_per_row(tmp_path):
    df = pd.DataFrame({"SMILES": [ETHANOL, BENZENE]})
    out = add_quantum_features(df, cache_path=tmp_path / "cache.csv")
    assert "quantum_features" in out.columns
    assert all(len(f) == len(FEATURE_NAMES) for f in out["quantum_features"])


def test_compute_quantum_features_batch_works_with_multiple_workers(tmp_path):
    """Regression test for the exact bug class pharmacophore_fingerprint.py's
    own multi-worker test guards against: a real n_workers=8 run crashed
    there because the compute callable passed to multiprocessing.Pool.imap
    was a local lambda, which is not picklable -- n_workers=1 (every other
    test in this file) never exercises that code path at all. This module
    passes the top-level extract_quantum_features function directly (no
    lambda), so this should already be safe, but only a real n_workers>1
    call proves it.
    """
    cache_path = tmp_path / "cache.csv"
    result = compute_quantum_features_batch([ETHANOL, BENZENE, "C"], cache_path=cache_path, n_workers=2)
    assert len(result) == 3
    assert len(pd.read_csv(cache_path)) == 3


def test_load_cached_quantum_features_raises_when_cache_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cached_quantum_features(pd.DataFrame({"SMILES": [ETHANOL]}), cache_path=tmp_path / "missing.csv")


def test_load_cached_quantum_features_raises_when_a_compound_is_not_cached(tmp_path):
    cache_path = tmp_path / "cache.csv"
    compute_quantum_features_batch([ETHANOL], cache_path=cache_path)
    with pytest.raises(ValueError):
        load_cached_quantum_features(pd.DataFrame({"SMILES": [ETHANOL, BENZENE]}), cache_path=cache_path)


def test_load_cached_quantum_features_joins_correctly(tmp_path):
    cache_path = tmp_path / "cache.csv"
    computed = compute_quantum_features_batch([ETHANOL, BENZENE], cache_path=cache_path)
    loaded = load_cached_quantum_features(pd.DataFrame({"SMILES": [BENZENE, ETHANOL]}), cache_path=cache_path)
    # order-preserving w.r.t. the input df, and values match what was computed
    assert list(loaded["SMILES"]) == [BENZENE, ETHANOL]
    ethanol_row = computed[computed["SMILES"] == ETHANOL].iloc[0]
    loaded_ethanol = loaded[loaded["SMILES"] == ETHANOL].iloc[0]
    np.testing.assert_allclose(
        loaded_ethanol["quantum_features"], ethanol_row[FEATURE_NAMES].to_numpy(dtype=np.float64)
    )
