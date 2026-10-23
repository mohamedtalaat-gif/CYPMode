# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pytest

pytest.importorskip("rdkit")
from rdkit import Chem

from cypmode.challenge import functional_groups as fg
from cypmode.challenge.pocket_pharmacophore import FAMILIES

KETOCONAZOLE = "CC(=O)N1CCN(CC1)c1ccc(OCC2COC(Cn3ccnc3)(O2)c2ccc(Cl)cc2Cl)cc1"
CASES = {
    "azole or pyridine (heme binder)": ("Cn1ccnc1", "CCO"),
    "basic amine": ("CCN(CC)CC", "CC(=O)NC"),
    "carboxylic acid": ("CC(=O)O", "CC(=O)OC"),
    "sulfonamide": ("CS(=O)(=O)NC", "CS(=O)(=O)C"),
    "amide": ("CC(=O)NC", "CC(=O)OC"),
    "urea or carbamate": ("CNC(=O)OC", "CC(=O)NC"),
    "ester": ("CC(=O)OC", "CC(=O)O"),
    "aryl halide": ("Clc1ccccc1", "CCCl"),
    "phenol": ("Oc1ccccc1", "CCO"),
    "aliphatic hydroxyl": ("CCO", "Oc1ccccc1"),
    "aryl ether": ("COc1ccccc1", "COC"),
    "nitrile": ("CC#N", "CC=N"),
    "nitro": ("C[N+](=O)[O-]", "CN"),
    "tert-butyl": ("CC(C)(C)c1ccccc1", "CC(C)c1ccccc1"),
    "methylenedioxyphenyl": ("c1cc2OCOc2cc1", "COc1ccccc1OC"),
    "furan": ("c1ccoc1", "c1ccsc1"),
    "thiophene": ("c1ccsc1", "c1ccoc1"),
    "terminal alkyne": ("CC#C", "CC#CC"),
}


def test_the_table_has_a_case_for_every_group_and_only_known_families():
    assert set(CASES) == set(fg.GROUPS)
    assert all(set(spec["families"]) <= set(FAMILIES) for spec in fg.GROUPS.values())
    assert all(len(patterns) and all(p is not None for p in patterns) for patterns in fg._PATTERNS.values())


@pytest.mark.parametrize("name", sorted(CASES))
def test_each_group_matches_its_example_and_not_its_counter_example(name):
    positive, negative = CASES[name]
    assert name in fg.groups_of(Chem.MolFromSmiles(positive)), name
    assert name not in fg.groups_of(Chem.MolFromSmiles(negative)), name


def test_ketoconazole_carries_its_azole_halides_amide_and_ether_and_each_match_lists_atoms():
    found = fg.groups_of(Chem.MolFromSmiles(KETOCONAZOLE))
    assert {"azole or pyridine (heme binder)", "aryl halide", "amide", "aryl ether"} <= set(found)
    assert len(found["aryl halide"]) == 2 and all(len(match) >= 2 for matches in found.values() for match in matches)
    assert "carboxylic acid" not in found and "basic amine" not in found


def test_the_matrix_flags_each_compound_and_an_unparseable_smiles_carries_nothing():
    matrix = fg.group_matrix(["CC(=O)O", "CCO", "not a smiles"])
    assert matrix.loc["CC(=O)O", "carboxylic acid"] and not matrix.loc["CCO", "carboxylic acid"] and not matrix.loc["not a smiles"].any()


def test_the_wilson_interval_stays_inside_zero_and_one_and_narrows_with_n():
    low, high = fg.wilson(0, 10)
    assert low == 0.0 or low < 1e-9
    assert 0 < high < 0.4
    narrow, wide = fg.wilson(50, 100), fg.wilson(5, 10)
    assert narrow[1] - narrow[0] < wide[1] - wide[0] and np.isnan(fg.wilson(0, 0)[0])


def test_the_rates_compare_the_carriers_with_the_rest_and_skip_rare_groups():
    smiles = ["CC(=O)O"] * 20 + ["CCO"] * 20 + ["c1ccoc1"] * 3
    labels = np.array([1] * 15 + [0] * 5 + [0] * 18 + [1] * 2 + [1, 1, 0])
    rates = fg.group_rates(fg.group_matrix(smiles), labels, min_compounds=10)
    acid = rates[rates.group == "carboxylic acid"].iloc[0]
    assert acid.n == 20 and acid.positives == 15 and acid.rate == pytest.approx(0.75) and acid.rate > acid.base_rate and acid.low < 0.75 < acid.high
    assert "furan" not in set(rates.group)  # only three carry it
    assert acid.rate_without == pytest.approx(labels[20:].mean())


def test_every_group_has_an_example_molecule_that_carries_it():
    assert set(fg.EXAMPLES) == set(fg.GROUPS)
    for name, smiles in fg.EXAMPLES.items():
        assert name in fg.groups_of(Chem.MolFromSmiles(smiles)), name


def test_a_pair_counts_for_the_group_it_gains_only_when_nothing_else_of_the_table_changes():
    smiles = ["c1ccccc1C", "c1ccccc1C(=O)O", "Oc1ccccc1C(=O)O", "Clc1ccccc1C"]
    pairs = (np.array([0, 0, 0]), np.array([1, 2, 3]), np.array([0, 1, 2]), ["a>>b", "c>>d", "e>>f"])
    table, detail = fg.group_gain_pairs(smiles, pairs, np.arange(4), np.array([0, 1, 1, 0]), min_pairs=1)
    columns = list(fg.group_matrix(smiles).columns)
    assert {columns[k] for k in detail["gained"] if k >= 0} == {"carboxylic acid", "aryl halide"}  # the pair that adds a phenol and an acid at once gains two groups and counts for none
    assert detail["gained"][1] == -1
    acid = table[table.group == "carboxylic acid"].iloc[0]
    assert acid.pairs == 1 and acid["mean"] == 1.0 and table[table.group == "aryl halide"].iloc[0]["mean"] == 0.0
    assert detail["delta"][detail["gained"] >= 0].tolist() == [1.0, 0.0]
