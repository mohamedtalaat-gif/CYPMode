"""GFN2-xTB part of the physics block. Own pytest invocation (tblite and LightGBM deadlock in one process):
    pytest tests/test_physics_block.py
"""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.tblite

pytest.importorskip("tblite")

from cypmode.challenge import physics_block as pb


@pytest.fixture(scope="module")
def mols():
    smiles = {"toluene": "Cc1ccccc1", "ethane": "CC", "tma": "CN(C)C", "pyridine": "c1ccncc1", "benzonitrile": "N#Cc1ccccc1", "anisole": "COc1ccccc1"}
    return {k: pb.compute_molecule(v) for k, v in smiles.items()}


def atoms(mols, name):
    return pd.DataFrame(mols[name][1])


def test_feature_vector_is_complete_and_atom_table_has_every_heavy_atom(mols):
    f, rows = mols["toluene"]
    assert set(f) == set(pb.FEATURE_NAMES) and len(rows) == 7
    assert np.isfinite(f["phys_ip_ev"]) and f["phys_ip_ev"] > f["phys_ea_ev"]
    assert f["phys_hardness_ev"] > 0 and f["phys_gap_ev"] > 0


def test_benzylic_hydrogen_is_easier_to_abstract_than_an_ethane_hydrogen(mols):
    assert atoms(mols, "toluene").hat_kcal.min() < atoms(mols, "ethane").hat_kcal.min()


def test_ionisation_potential_orders_electron_rich_below_electron_poor(mols):
    ip = {k: v[0]["phys_ip_ev"] for k, v in mols.items()}
    assert ip["tma"] < ip["toluene"] < ip["benzonitrile"] and ip["anisole"] < ip["pyridine"]


def test_proton_affinity_orders_basic_nitrogens_above_nitrile(mols):
    pa = {k: atoms(mols, k).pa_gas_kcal.max() for k in ("tma", "pyridine", "benzonitrile")}
    assert pa["pyridine"] > pa["benzonitrile"] and pa["tma"] > pa["benzonitrile"]
    assert atoms(mols, "pyridine").pa_water_kcal.max() > 0


def test_fukui_minus_is_largest_on_the_lone_pair_nitrogen_of_an_amine(mols):
    a = atoms(mols, "tma")
    assert a.loc[a.symbol == "N", "f_minus"].iloc[0] == a.f_minus.max()


def test_extraction_is_deterministic_and_resumable(tmp_path):
    smiles = ["CCO", "Cc1ccccc1", "c1ccncc1"]
    cache, atoms_path = tmp_path / "c.csv", tmp_path / "a.csv.gz"
    pb.extract(smiles, cache, atoms_path, n_workers=1)
    first = pd.read_csv(cache)
    pb.extract(smiles, cache, atoms_path, n_workers=1)  # nothing left to do
    assert len(pd.read_csv(cache)) == 3 and len(pd.read_csv(atoms_path)) == 3 + 7 + 6
    again = pb.compute_molecule("Cc1ccccc1")[0]
    row = first[first.SMILES == pb.canonical("Cc1ccccc1")].iloc[0]
    for name in ("phys_ip_ev", "phys_homo_ev", "phys_benzylic_hat_min"):
        assert row[name] == pytest.approx(again[name], rel=1e-9)
    assert pb.load_physics_features(smiles, cache).shape == (3, len(pb.FEATURE_NAMES))


def test_unembeddable_or_unparseable_input_gives_a_nan_row_not_a_crash():
    f, rows = pb.compute_molecule("not a smiles !!!")
    assert rows == [] and all(np.isnan(v) for v in f.values())
