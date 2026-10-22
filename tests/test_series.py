# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pytest

from cypmode.validation import series as sr
from cypmode.validation import series_consensus as sc


def test_common_core_of_two_analogs_covers_the_shared_scaffold():
    core = sr.common_core("Cc1ccc(CCN)cc1", "Clc1ccc(CCN)cc1")
    assert len(core) >= 8 and len({a for a, _ in core}) == len(core)
    assert sr.common_core("CCO", "c1ccccc1") == []  # nothing shared worth a core


def test_rotation_superposes_a_rigidly_moved_point_set():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.linalg.det(q))
    moved = x @ q.T + np.array([5.0, -2.0, 1.0])
    r = sr._rotation(moved, x)
    back = (moved - moved.mean(axis=0)) @ r + x.mean(axis=0)
    assert np.allclose(back, x, atol=1e-8)


def test_combined_pick_prefers_the_pose_that_both_agrees_with_the_pool_and_with_the_analog():
    mbr = np.array([0.6, 0.7, 0.5, 0.4])
    consensus = np.array([4.0, 1.0, 0.8, 5.0])
    assert sc.combined_pick(mbr, consensus) == 1
    assert sc.combined_pick(mbr, np.full(4, np.nan)) == 1  # without a partner the pool consensus decides


def test_series_benchmark_pairs_use_distinct_compounds_and_exclude_the_existing_benchmark():
    import json
    from pathlib import Path

    path = Path("data/structure_track/inputs/cyp3a4_reference_ligands.json")
    if not path.exists():
        pytest.skip("reference set not present")
    refs = json.loads(path.read_text())["ligands"]
    t = sr.select_series_benchmark(refs, set(), 3)
    assert len(t) == 6 and t.Molecule_Name.is_unique and sorted(t.series.unique()) == [1, 2, 3]
    from cypmode.validation.pose_rescoring import connectivity_key

    for _, pair in t.groupby("series"):
        a, b = pair.SMILES
        assert connectivity_key(a) != connectivity_key(b)


def test_official_pairs_are_symmetric_free_and_above_the_threshold():
    pairs = sc.official_pairs()
    assert all(s >= 0.5 for _, _, s in pairs) and len({(a, b) for a, b, _ in pairs}) == len(pairs)
