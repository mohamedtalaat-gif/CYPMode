"""Chemistry bookkeeping of the physics block that needs no quantum calculation (safe next to LightGBM)."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cypmode.challenge import physics_block as pb


def roles(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return mol, *pb.assign_roles(mol)


def test_ch_roles_follow_the_chemistry():
    _, ch, _ = roles("Cc1ccccc1")
    assert ch[0] == "benzylic" and ch[1] == "" and ch[2] == "aromatic_ch"
    _, ch, _ = roles("COc1ccccc1")
    assert ch[0] == "alpha_O"
    _, ch, _ = roles("CN(C)CCC")
    assert ch[0] == "alpha_N" and ch[4] == "aliphatic"
    _, ch, _ = roles("c1ccc2OCOc2c1")
    assert "methylenedioxy" in ch


def test_binder_roles_distinguish_pyridine_azole_amine_and_leave_pyrrole_nitrogen_alone():
    _, _, b = roles("c1ccncc1")
    assert b[3] == "pyridine_N"
    _, _, b = roles("c1c[nH]cn1")
    assert b.count("azole_N") == 1 and b[2] == ""  # the N-H nitrogen has no free lone pair for the iron
    _, _, b = roles("CN(C)C")
    assert b[1] == "amine_N"
    _, _, b = roles("CC(=O)N(C)C")
    assert "amine_N" not in b  # an amide nitrogen does not coordinate
    _, _, b = roles("N#Cc1ccccc1")
    assert b[0] == "nitrile_N"
    _, _, b = roles("c1ccsc1")
    assert "thiophene_S" in b


def test_roles_are_disjoint_first_match_wins():
    _, ch, _ = roles("COCc1ccccc1")  # the CH2 is both benzylic and alpha-O: alpha-O is listed first
    assert ch[2] == "alpha_O"


def test_proton_sits_on_the_lone_pair_at_n_h_distance():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccncc1"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    xyz = mol.GetConformer().GetPositions()
    n = 3
    neighbours = [a.GetIdx() for a in mol.GetAtomWithIdx(n).GetNeighbors()]
    h = pb.proton_position(xyz, n, neighbours)
    assert np.linalg.norm(h - xyz[n]) == pytest.approx(pb.N_H_BOND_ANGSTROM, abs=1e-6)
    bisector = (xyz[neighbours[0]] + xyz[neighbours[1]]) / 2 - xyz[n]
    cosine = np.dot(h - xyz[n], bisector) / (np.linalg.norm(h - xyz[n]) * np.linalg.norm(bisector))
    assert cosine == pytest.approx(-1.0, abs=1e-6)  # exactly opposite the two ring bonds


def test_pharmacophore_geometry_measures_binder_to_remote_site_and_flags_adjacent_sites():
    mol = Chem.AddHs(Chem.MolFromSmiles("CN(C)CCCCc1ccccc1"))
    ids, energies = pb.conformer_ensemble(mol, n=4)
    ch, binder = pb.assign_roles(mol)
    n_atom = binder.index("amine_N")
    benzylic = ch.index("benzylic")
    alpha = ch.index("alpha_N")
    hat = {alpha: 118.0, benzylic: 121.0}  # the alpha-N hydrogen is weakest, the benzylic one is the best remote site
    out = pb.pharmacophore_geometry(mol, ids, energies, (ch, binder), hat, ids[int(np.argmin(energies))])
    assert out["n_binders"] == 1 and out["best_is_adjacent"] == 1.0
    assert out["hat_remote_minus_best"] == pytest.approx(3.0)
    topological = Chem.GetDistanceMatrix(mol)[n_atom, benzylic]
    assert 3.0 <= out["d_binder_remote_mean"] <= 1.54 * topological  # shorter than the fully extended chain
    assert out["d_binder_remote_min"] <= out["d_binder_remote_mean"] <= out["d_binder_remote_max"]
    assert out["d_amine_remote_mean"] == pytest.approx(out["d_binder_remote_mean"])
    assert out["n_rings_within_7A"] >= 0


def test_pharmacophore_geometry_without_binders_or_sites_is_all_nan_but_complete():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    ids, energies = pb.conformer_ensemble(mol, n=2)
    ch, binder = pb.assign_roles(mol)
    out = pb.pharmacophore_geometry(mol, ids, energies, (ch, binder), {}, ids[0])
    assert set(out) == set(pb.PHARMACOPHORE_FEATURES) and out["n_binders"] == 0.0


def test_feature_names_are_unique_and_prefixed():
    assert len(pb.FEATURE_NAMES) == len(set(pb.FEATURE_NAMES)) and all(n.startswith("phys_") for n in pb.FEATURE_NAMES)


def test_load_physics_features_refuses_a_missing_cache_and_missing_compounds(tmp_path):
    with pytest.raises(FileNotFoundError, match="--extract"):
        pb.load_physics_features(["CCO"], tmp_path / "none.csv")
    import pandas as pd

    row = {"SMILES": pb.canonical("OCC"), **{n: 1.0 for n in pb.FEATURE_NAMES}}
    pd.DataFrame([row]).to_csv(tmp_path / "c.csv", index=False)
    assert pb.load_physics_features(["CCO"], tmp_path / "c.csv").shape == (1, len(pb.FEATURE_NAMES))  # canonicalised lookup
    with pytest.raises(ValueError, match="no physics features"):
        pb.load_physics_features(["CCN"], tmp_path / "c.csv")
