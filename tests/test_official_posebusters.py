import zipfile
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cypmode.validation import final_recipe as fr
from cypmode.validation.official_posebusters import (
    MARGIN_FLOOR,
    OfficialReport,
    bond_orders_from_template,
)

ZIP = Path("data/structure_track/outputs/cyp3a4/final/structures_final.zip")


def _coordinates_without_bond_orders(smiles: str, with_hydrogens: bool) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    if not with_hydrogens:
        mol = Chem.RemoveHs(mol)
    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True) if with_hydrogens else atom.SetNoImplicit(False)
    return rw.GetMol()


@pytest.mark.parametrize("with_hydrogens", [False, True])
def test_bond_orders_come_back_from_the_template(with_hydrogens):
    smiles = "Cc1ccc(C(=O)N2CCOCC2)cn1"
    template = Chem.MolFromSmiles(smiles)
    bare = _coordinates_without_bond_orders(smiles, with_hydrogens)
    assigned = bond_orders_from_template(template, bare)
    assert Chem.MolToSmiles(Chem.RemoveHs(assigned)) == Chem.MolToSmiles(template)
    assert assigned.GetNumAtoms() == bare.GetNumAtoms()
    assert np.allclose(assigned.GetConformer().GetPositions(), bare.GetConformer().GetPositions())


def test_a_template_that_does_not_match_is_refused():
    bare = _coordinates_without_bond_orders("Cc1ccc(C(=O)N2CCOCC2)cn1", False)
    with pytest.raises(ValueError):
        bond_orders_from_template(Chem.MolFromSmiles("ClCCO"), bare)


def test_a_pose_is_safe_only_when_it_passes_and_clears_the_margin_floor():
    assert OfficialReport(passed=True, distances={"protein": MARGIN_FLOOR + 0.01, "organic_cofactors": 0.9}).safe
    assert not OfficialReport(passed=True, distances={"protein": 0.9, "organic_cofactors": MARGIN_FLOOR - 0.01}).safe
    assert not OfficialReport(passed=False, distances={"protein": 0.9}).safe
    assert not OfficialReport(passed=True).safe  # no distance could be computed


def test_margin_candidates_prefer_safe_valid_poses_and_fall_back_to_the_widest_margin():
    ok = np.array([True, True, True, False])
    margins = np.array([0.76, 0.85, 0.81, 0.95])
    assert fr.margin_candidates([0, 1, 2], ok, margins) == [1, 2]  # the poses below the floor are dropped, the order is kept
    assert fr.margin_candidates([0, 1, 2, 3], ok, margins) == [1, 2]  # an invalid pose never enters, however wide its margin
    assert fr.margin_candidates([0, 2], ok, np.array([0.76, 0.0, 0.78, 0.0])) == [2]  # none is safe: the widest margin alone
    assert fr.margin_candidates([3], ok, margins) == [3]  # nothing valid: unchanged


def test_select_pose_skips_a_thin_pose_that_the_model_ranked_first():
    pool = [("run", "boltz", Path(f"p{i}.cif")) for i in range(3)]
    margin = {"p0.cif": 0.755, "p1.cif": 0.83, "p2.cif": 0.9}
    best, ok, _ = fr.select_pose(pool, "first_valid", lambda p: True, margin=lambda p: margin[p.name])
    assert (best, ok) == (1, True)
    assert fr.select_pose(pool, "first_valid", lambda p: True)[0] == 0  # without margins the old rule holds


@pytest.mark.skipif(not ZIP.exists(), reason="requires the final structure zip")
def test_the_scorer_exact_call_passes_a_submitted_pose_with_all_22_checks(tmp_path):
    import pandas as pd

    from cypmode.validation.official_posebusters import check_pdb

    official = pd.read_csv("data/structure_track/inputs/structure_track_test_blinded.csv")
    name = official.Molecule_Name.iloc[0]
    with zipfile.ZipFile(ZIP) as zf:
        pdb = tmp_path / f"{name}.pdb"
        pdb.write_bytes(zf.read(f"{name}.pdb"))
    report = check_pdb(pdb, official.SMILES.iloc[0])
    assert report.passed and report.n_checks == 22 and report.n_copies == 1
    assert 0.75 < report.distance_margin < 1.5
