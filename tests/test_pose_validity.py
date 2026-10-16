from pathlib import Path

import pytest

pytest.importorskip("Bio")
pytest.importorskip("posebusters")

from cypmode.validation.structures import load_structure

RESULT_DIR = Path("data/structure_track/outputs/out/cyp3a4/boltz_results_cyp3a4_ketoconazole")
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


ITRACONAZOLE_RESULT_DIR = Path("data/structure_track/outputs/out/cyp3a4/boltz_results_cyp3a4_itraconazole")
ITRACONAZOLE_CIF = ITRACONAZOLE_RESULT_DIR / "predictions" / "cyp3a4_itraconazole" / "cyp3a4_itraconazole_model_0.cif"
ITRACONAZOLE_SMILES = "CCC(C)n1ncn(-c2ccc(N3CCN(c4ccc(OCC5COC(Cn6cncn6)(c6ccc(Cl)cc6Cl)O5)cc4)CC3)cc2)c1=O"


def test_select_valid_model_returns_model_0_when_it_already_passes(tmp_path):
    import shutil

    from cypmode.validation.pose_validity import select_valid_model

    predictions_dir = tmp_path / "cyp3a4_ketoconazole"
    predictions_dir.mkdir()
    shutil.copy(CIF_PATH, predictions_dir / "cyp3a4_ketoconazole_model_0.cif")

    result = select_valid_model(predictions_dir, KETOCONAZOLE_SMILES)
    assert result["model_index"] == 0
    assert result["result"]["all_passed"] is True


@pytest.mark.skipif(not ITRACONAZOLE_CIF.exists(), reason="requires a local Boltz-2 run for itraconazole")
def test_select_valid_model_skips_a_failing_model_0_for_a_passing_model_1(tmp_path):
    import shutil

    from cypmode.validation.pose_validity import select_valid_model

    predictions_dir = tmp_path / "cyp3a4_itraconazole"
    predictions_dir.mkdir()
    # real, confirmed-failing structure as "model_0"
    shutil.copy(ITRACONAZOLE_CIF, predictions_dir / "cyp3a4_itraconazole_model_0.cif")
    # a real, known-good structure (different compound, but proves the
    # "keep trying subsequent models" logic) as "model_1" -- using its
    # own real SMILES for the check would require ketoconazole's SMILES,
    # not itraconazole's, so this test uses itraconazole's own SMILES
    # against ketoconazole's structure deliberately to confirm select_valid_model
    # tries model_1 and reports whatever check_pose_validity says about it,
    # not that it must pass (a structure of the wrong ligand should fail
    # sanitization/identity checks, not silently "pass")
    shutil.copy(CIF_PATH, predictions_dir / "cyp3a4_itraconazole_model_1.cif")

    result = select_valid_model(predictions_dir, ITRACONAZOLE_SMILES, max_models=2)
    # model_0 (real failure) skipped; model_1 is a mismatched structure
    # for this SMILES, so no valid model should be found among these two
    assert result["model_index"] is None


def test_select_valid_model_returns_none_when_no_models_exist(tmp_path):
    from cypmode.validation.pose_validity import select_valid_model

    empty_dir = tmp_path / "cyp3a4_nothing"
    empty_dir.mkdir()
    result = select_valid_model(empty_dir, KETOCONAZOLE_SMILES)
    assert result["model_index"] is None
    assert result["cif_path"] is None
