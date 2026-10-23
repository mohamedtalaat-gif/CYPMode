# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")

from cypmode.challenge import pocket_pharmacophore as pp
from cypmode.validation import official_pharmacophore as official


def _features(**by_family):
    out = {f: np.zeros((0, 3)) for f in pp.FAMILIES}
    out.update({k: np.array(v, dtype=float).reshape(-1, 3) for k, v in by_family.items()})
    return out


def test_the_sites_that_a_pose_matches_are_listed_with_their_family_weight_and_position():
    sites = {f: np.zeros((0, 4)) for f in pp.FAMILIES}
    sites["Acceptor"] = np.array([[0.0, 0.0, 6.0, 1.0], [8.0, 0.0, 6.0, 0.5]])
    table = official.matched_sites(_features(Acceptor=[[0.5, 0.0, 6.0]]), sites)
    assert table.matched.tolist() == [True, False] and table.weight.tolist() == [1.0, 0.5] and set(table.family) == {"Acceptor"}


def test_the_common_sites_need_enough_ligands_with_a_feature_of_the_family_at_the_same_place():
    together = {f"L{i}": _features(Aromatic=[[1.0 + 0.3 * (i % 3), 0.0, 6.0]]) for i in range(16)}
    scattered = {f"S{i}": _features(Aromatic=[[-10.0 + 2.0 * i, 8.0, 12.0]]) for i in range(4)}
    sites = official.common_sites({**together, **scattered}, minimum=14)
    assert len(sites) == 1 and sites.family[0] == "Aromatic" and sites.ligands[0] >= 14 and abs(sites.x[0] - 1.3) < 1.5 and abs(sites.z[0] - 6.0) < 1.0
    assert official.common_sites(scattered, minimum=14).empty


def test_the_similarity_of_two_ligands_with_the_same_features_is_one_and_the_order_groups_the_alike():
    a = _features(Donor=[[0.0, 0.0, 6.0]], Aromatic=[[3.0, 0.0, 8.0]])
    b = _features(Donor=[[0.0, 0.0, 6.0]], Aromatic=[[3.0, 0.0, 8.0]])
    c = _features(Hydrophobe=[[-8.0, 8.0, 12.0]])
    matrix = official.similarity({"a": a, "c": c, "b": b})
    assert matrix.loc["a", "b"] == pytest.approx(1.0) and matrix.loc["a", "c"] < 0.2
    order = list(matrix.index)
    assert abs(order.index("a") - order.index("b")) == 1


def test_the_spread_of_the_pharmacophore_across_the_pool_is_the_median_distance_to_the_nearest_feature_of_the_family():
    final = _features(Donor=[[0.0, 0.0, 6.0]])
    others = [_features(Donor=[[1.0, 0.0, 6.0]]), _features(Donor=[[0.0, 3.0, 6.0]]), _features()]
    assert official.feature_spread(final, others) == pytest.approx(2.0)
    assert np.isnan(official.feature_spread(final, [_features()]))
    assert official.same_features(final, _features(Donor=[[0.0, 0.0, 6.005]])) and not official.same_features(final, others[0])


def test_the_receptor_side_finds_the_partner_of_a_polar_feature_flags_a_buried_one_without_a_partner_and_lists_the_contacts():
    ring = [(2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (-2.0, 0.0, 0.0), (0.0, -2.0, 0.0)]
    rows = [(0.0, 0.0, 0.0, "FE", "HEM500", "FE"), *[(*p, "N", "HEM500", n) for p, n in zip(ring, ("NA", "NB", "NC", "ND"), strict=True)]]
    rows += [(0.0, 3.0, 8.0, "O", "SER119", "OG"), (6.0, 6.0, 8.0, "C", "PHE304", "CZ")]
    rows += [(float(10 + i % 5), float(i // 5), 14.0, "C", f"LEU{i}", "CD1") for i in range(40)]  # a dense shell around the point (12, 2, 14)
    atoms = pd.DataFrame(rows, columns=["x", "y", "z", "element", "residue", "atom"])
    table = official.receptor_side(_features(Donor=[[0.0, 1.0, 8.0], [12.0, 2.0, 14.0]], Acceptor=[[0.0, 0.0, 2.3]], Aromatic=[[6.0, 4.0, 8.0]]), atoms)
    donors = table[table.family == "Donor"].reset_index(drop=True)
    assert donors.partner[0] == "SER119:OG" and donors.partner_distance[0] == pytest.approx(2.0) and not donors.unsatisfied[0]
    assert donors.buried[1] and donors.partner[1] == "" and donors.unsatisfied[1]
    assert table[table.family == "Acceptor"].partner.iloc[0] == "HEM500:FE"  # a heme iron within the distance is the partner of an acceptor
    assert "PHE304" in table[table.family == "Aromatic"].contacts.iloc[0]


def test_the_valid_poses_of_the_pool_of_a_ligand_are_the_existing_files_that_passed(tmp_path):
    good = tmp_path / "a.cif"
    good.write_text("data_x\n")
    table = pd.DataFrame({"ligand": ["L1", "L1", "L1", "L2"], "passed": [True, False, True, True], "cif": [str(good), str(good), str(tmp_path / "missing.cif"), str(good)]})
    assert official.pool_poses(table, "L1") == [good] and official.pool_poses(table, "L3") == []
