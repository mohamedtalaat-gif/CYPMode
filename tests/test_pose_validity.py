from pathlib import Path

import pytest

pytest.importorskip("Bio")
pytest.importorskip("posebusters")

from cypmode.validation.structures import load_structure

RESULT_DIR = Path("data/boltz_test/out/boltz_results_cyp3a4_ketoconazole")
CIF_PATH = RESULT_DIR / "predictions" / "cyp3a4_ketoconazole" / "cyp3a4_ketoconazole_model_0.cif"
KETOCONAZOLE_SMILES = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"

pytestmark = pytest.mark.skipif(
    not CIF_PATH.exists(),
    reason="requires a local Boltz-2 run; see README.md's Validation section",
)


def test_check_pose_validity_passes_a_real_known_good_structure():
    # Real, confirmed result: ketoconazole's real predicted structure
    # passes every PoseBusters check.
    from cypmode.validation.pose_validity import check_pose_validity

    structure = load_structure(CIF_PATH)
    result = check_pose_validity(structure, KETOCONAZOLE_SMILES)
    assert result["all_passed"] is True
    assert len(result["checks"]) > 15  # PoseBusters runs ~20 real checks
    assert all(isinstance(v, bool) for v in result["checks"].values())


def test_check_pose_validity_does_not_mutate_the_input_structure():
    from cypmode.validation.pose_validity import check_pose_validity
    from cypmode.validation.structures import _ligand_atoms

    structure = load_structure(CIF_PATH)
    original_resname = _ligand_atoms(structure)[0].get_parent().resname
    check_pose_validity(structure, KETOCONAZOLE_SMILES)
    assert _ligand_atoms(structure)[0].get_parent().resname == original_resname


def test_check_pose_validity_reports_a_bond_length_check():
    from cypmode.validation.pose_validity import check_pose_validity

    structure = load_structure(CIF_PATH)
    result = check_pose_validity(structure, KETOCONAZOLE_SMILES)
    assert "bond_lengths" in result["checks"]
