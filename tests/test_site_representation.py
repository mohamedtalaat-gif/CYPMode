# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd
import pytest

from cypmode.validation import site_representation as site


def _rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_the_pocket_rmsd_of_a_rigid_copy_is_zero_and_that_of_a_displaced_residue_is_not():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(60, 3)) * 8
    moved = base @ _rotation(0.7).T + np.array([5.0, -3.0, 2.0])
    assert site.kabsch_rmsd(base, moved) < 1e-6
    bent = moved.copy()
    bent[:10] += 3.0
    assert 0.5 < site.kabsch_rmsd(base, bent) < 3.0


def test_the_core_residues_are_those_that_enough_pockets_have_and_a_pair_needs_enough_of_them_in_common():
    pockets = [{r: np.zeros(3) for r in range(100) if r != 5}] * 9 + [{r: np.zeros(3) for r in range(100) if r != 7}]
    core = site.core_residues(pockets)
    assert 7 in core and 5 not in core and len(core) == 99
    few = {r: np.zeros(3) for r in range(10)}
    assert np.isnan(site.pocket_rmsd(few, few, set(range(10)))[0])


def test_the_nearest_other_copy_skips_the_same_entry_and_the_same_ligand():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(80, 3)) * 8
    make = lambda shift: {r: base[r] + shift for r in range(80)}
    pockets = [make(0.0), make(0.2), make(0.4), make(3.0)]
    ids = ["A", "A", "B", "C"]
    keys = ["k1", "k2", "k1", "k4"]
    core = set(range(80))
    # copy 0: copy 1 has the same entry, copy 2 the same ligand, so only copy 3 counts
    value = site.nearest_other(0, pockets, keys, ids, core)
    assert value == pytest.approx(site.pocket_rmsd(pockets[0], pockets[3], core)[0])
    assert np.isnan(site.nearest_other(0, pockets[:3], keys[:3], ids[:3], core))


def test_a_predicted_pocket_is_compared_with_the_references_of_other_compounds_only():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(80, 3)) * 8
    near = {r: base[r] for r in range(80)}
    far = {r: base[r] + rng.normal(size=3) * 4 for r in range(80)}
    value, nearest = site.nearest_reference(near, [near, far], ["own", "other"], "own", set(range(80)))
    assert nearest == 1 and value == pytest.approx(site.pocket_rmsd(near, far, set(range(80)))[0])


def test_the_percentile_flag_and_the_correlations_ignore_missing_values():
    assert site.percentile(pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])) == pytest.approx(np.percentile([1, 2, 3, 4], 95))
    table = pd.DataFrame({"compound": ["a"] * 6 + ["b"] * 6, "true_rmsd": list(range(6)) * 2, "x": [*range(6), *range(5, -1, -1)]})
    pooled, per = site.correlation(table, "x")
    assert abs(pooled) < 0.2 and per == pytest.approx(0.0, abs=1e-9)


def test_the_summary_reports_the_flag_the_cut_and_the_shares_beyond_them():
    crystal = pd.DataFrame({"pocket_nearest_rmsd": [0.5, 0.6, 0.7, 2.0], "outside": [0.0, 0.0, 0.1, 0.3]})
    predicted = pd.DataFrame({"compound": ["a"] * 4, "true_rmsd": [1.0, 2.0, 3.0, 4.0], "pocket_nearest_rmsd": [0.5, 0.9, 1.5, 3.0], "beyond_pocket_flag": [False, False, True, True], "outside": [0.0, 0.1, 0.2, 0.4], "beyond_support_cut": [False, False, True, True]})
    lines = site.summarize(crystal, predicted, flag=1.8, cut=0.27, n_core=64)
    assert "64 core residues" in lines[0] and "50% beyond the flag" in lines[1] and "predicted poses beyond the cut: 50% of 4" in lines[3] and lines[4].startswith("SITE CHECK flags")
