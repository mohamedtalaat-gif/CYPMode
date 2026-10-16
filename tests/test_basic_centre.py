import numpy as np
import pandas as pd
import pytest

pytest.importorskip("dimorphite_dl")

from cypmode.challenge import basic_centre as bc

C = {name: i for i, name in enumerate(bc.COLUMNS)}


def test_the_row_has_one_value_per_column_for_every_kind_of_molecule():
    for smiles in ("CN1CCC(CC1)c1ccccc1", "CCO", "NC(=N)c1ccccc1", "O=[N+]([O-])c1ccccc1", "CC(C)C"):
        row = bc.basic_centre_row(smiles)
        assert len(row) == len(bc.COLUMNS) and np.isfinite(row).all()


def test_a_tertiary_amine_is_a_basic_centre_a_few_angstrom_from_its_ring():
    row = bc.basic_centre_row("CN1CCC(CC1)c1ccccc1")
    assert row[C["has_basic_centre"]] == 1 and row[C["n_tertiary"]] == 1 and row[C["share_charged_states"]] == 1
    assert row[C["path_min_to_aromatic"]] == 4  # N-C-C-C(ring)-aromatic
    assert 4.5 <= row[C["ring_nearest_mean"]] <= 6.5 and row[C["share_in_window"]] > 0.5


def test_other_cation_classes_are_told_apart():
    assert bc.basic_centre_row("NC(=N)c1ccccc1")[C["n_amidinium"]] == 1
    assert bc.basic_centre_row("C[N+](C)(C)Cc1ccccc1")[C["n_quaternary"]] == 1
    assert bc.basic_centre_row("CCCN")[C["n_primary"]] == 1


def test_nitro_groups_and_neutral_molecules_have_no_basic_centre():
    for smiles in ("O=[N+]([O-])c1ccccc1", "CCO", "c1ccccc1C(=O)O"):
        row = bc.basic_centre_row(smiles)
        assert row[C["has_basic_centre"]] == 0 and row[C["n_basic_centres"]] == 0 and row[C["geometry_ok"]] == 0


def test_a_basic_centre_without_a_ring_marks_the_ring_columns_missing():
    row = bc.basic_centre_row("CCN(CC)CC")
    assert row[C["has_basic_centre"]] == 1 and row[C["n_aromatic_rings"]] == 0 and row[C["ring_nearest_mean"]] == -1


def test_the_cache_is_keyed_by_canonical_smiles_and_read_back_in_order(tmp_path):
    path = tmp_path / "bc.csv"
    frame = bc.extract(["OCC", "CN1CCC(CC1)c1ccccc1"], path)
    assert list(frame.index) == ["CCO", "CN1CCC(c2ccccc2)CC1"]
    block = bc.load_basic_centre_features(["CN1CCC(c2ccccc2)CC1", "CCO"], path)
    assert block.shape == (2, len(bc.COLUMNS)) and block[0, C["has_basic_centre"]] == 1 and block[1, C["has_basic_centre"]] == 0
    with pytest.raises(KeyError):
        bc.load_basic_centre_features(["CCCl"], path)
    with pytest.raises(FileNotFoundError):
        bc.load_basic_centre_features(["CCO"], tmp_path / "missing.csv")
