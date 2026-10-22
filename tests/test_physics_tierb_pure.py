# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from cypmode.challenge import physics_block as pb
from cypmode.challenge import physics_tierb as tb


def test_boltzmann_weights_sum_to_one_and_favour_the_lowest_energy():
    w = tb.boltzmann_weights(np.array([0.0, 0.6, 3.0]))
    assert w.sum() == pytest.approx(1.0) and w[0] > w[1] > w[2]
    assert tb.boltzmann_weights(np.array([5.0]))[0] == pytest.approx(1.0)


def test_mobile_region_is_the_centre_its_neighbours_and_their_hydrogens():
    mol = Chem.AddHs(Chem.MolFromSmiles("Cc1ccccc1"))
    region = tb.mobile_atoms(mol, 0, shell=1)
    assert 0 in region and 1 in region and all(h.GetIdx() in region for h in mol.GetAtomWithIdx(0).GetNeighbors())
    assert len(tb.mobile_atoms(mol, 0, shell=2)) > len(region)


def test_top_sites_orders_by_vertical_energy_and_skips_atoms_without_one():
    atoms = pd.DataFrame({"SMILES": ["a"] * 4, "atom": [0, 1, 2, 3], "hat_kcal": [120.0, np.nan, 110.0, 115.0], "ch_role": ["x", "", "benzylic", "alpha_N"]})
    assert tb.top_sites(atoms, 2)["a"] == [(2, 110.0, "benzylic"), (3, 115.0, "alpha_N")]


def test_loader_refuses_missing_cache_and_missing_compounds(tmp_path):
    with pytest.raises(FileNotFoundError, match="--extract"):
        tb.load_tierb_features(["CCO"], tmp_path / "none.csv")
    pd.DataFrame([{"SMILES": pb.canonical("OCC"), **{n: 1.0 for n in tb.FEATURE_NAMES}}]).to_csv(tmp_path / "c.csv", index=False)
    assert tb.load_tierb_features(["CCO"], tmp_path / "c.csv").shape == (1, len(tb.FEATURE_NAMES))
    with pytest.raises(ValueError, match="no tier-B"):
        tb.load_tierb_features(["CCN"], tmp_path / "c.csv")


def test_feature_names_are_unique_and_prefixed():
    assert len(set(tb.FEATURE_NAMES)) == len(tb.FEATURE_NAMES) and all(n.startswith("phys_b_") for n in tb.FEATURE_NAMES)
