from pathlib import Path

import pytest

pytest.importorskip("Bio")

from cypmode.validation.structures import load_structure

RESULT_DIR = Path("data/structure_track/outputs/out/cyp3a4/boltz_results_cyp3a4_ketoconazole")
CIF_PATH = RESULT_DIR / "predictions" / "cyp3a4_ketoconazole" / "cyp3a4_ketoconazole_model_0.cif"

pytestmark = pytest.mark.skipif(
    not CIF_PATH.exists(),
    reason="requires a local Boltz-2 run; see README.md's Validation section",
)


def test_ligand_burial_sasa_matches_a_known_deeply_buried_compound():
    from cypmode.validation.pocket_geometry import ligand_burial_sasa

    # Real, known result for ketoconazole (confirmed directly): sits deep
    # in the CYP3A4 active site coordinating the heme iron -- expect high
    # burial, not a loosely-bound surface pose.
    structure = load_structure(CIF_PATH)
    result = ligand_burial_sasa(structure)
    assert result["bound_sasa_A2"] < result["unbound_sasa_A2"]
    assert 0.5 < result["burial_fraction"] < 1.0


def test_ligand_burial_sasa_does_not_mutate_the_input_structure():
    structure = load_structure(CIF_PATH)
    from cypmode.validation.structures import _ligand_atoms

    original_resname = _ligand_atoms(structure)[0].get_parent().resname
    from cypmode.validation.pocket_geometry import ligand_burial_sasa

    ligand_burial_sasa(structure)
    assert _ligand_atoms(structure)[0].get_parent().resname == original_resname


def test_pocket_volume_returns_a_real_positive_volume():
    pytest.importorskip("pyKVFinder")
    from cypmode.validation.pocket_geometry import pocket_volume

    structure = load_structure(CIF_PATH)
    result = pocket_volume(structure)
    assert result["pocket_volume_A3"] > 0
    assert result["pocket_area_A2"] > 0
    assert result["n_cavities_detected"] >= 1


def test_pocket_volume_does_not_mutate_the_input_structure():
    pytest.importorskip("pyKVFinder")
    from cypmode.validation.pocket_geometry import pocket_volume
    from cypmode.validation.structures import _ligand_atoms

    structure = load_structure(CIF_PATH)
    original_resname = _ligand_atoms(structure)[0].get_parent().resname
    pocket_volume(structure)
    assert _ligand_atoms(structure)[0].get_parent().resname == original_resname
