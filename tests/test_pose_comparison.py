import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("Bio")

from cypmode.validation.structures import _ligand_mol_3d, load_structure

RESULT_DIR = Path("data/structure_track/outputs/out/cyp3a4/boltz_results_cyp3a4_ketoconazole")
CIF_PATH = RESULT_DIR / "predictions" / "cyp3a4_ketoconazole" / "cyp3a4_ketoconazole_model_0.cif"
KETOCONAZOLE_SMILES = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"

pytestmark = pytest.mark.skipif(
    not CIF_PATH.exists(),
    reason="requires a local Boltz-2 run; see README.md's Validation section",
)


def test_ligand_pose_rmsd_is_zero_for_an_identical_pose():
    from cypmode.validation.pose_comparison import ligand_pose_rmsd

    structure = load_structure(CIF_PATH)
    mol, _ = _ligand_mol_3d(structure, KETOCONAZOLE_SMILES)
    rmsd = ligand_pose_rmsd(copy.deepcopy(mol), copy.deepcopy(mol))
    assert rmsd == pytest.approx(0.0, abs=1e-6)


def test_ligand_pose_rmsd_ignores_pure_rigid_body_translation():
    from cypmode.validation.pose_comparison import ligand_pose_rmsd

    structure = load_structure(CIF_PATH)
    mol, _ = _ligand_mol_3d(structure, KETOCONAZOLE_SMILES)
    shifted = copy.deepcopy(mol)
    conf = shifted.GetConformer()
    for i in range(shifted.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (pos.x + 5.0, pos.y, pos.z))

    rmsd = ligand_pose_rmsd(copy.deepcopy(mol), shifted)
    assert rmsd == pytest.approx(0.0, abs=1e-6)  # alignment removes a pure translation


def test_ligand_pose_rmsd_detects_a_real_conformational_difference():
    from cypmode.validation.pose_comparison import ligand_pose_rmsd

    structure = load_structure(CIF_PATH)
    mol, _ = _ligand_mol_3d(structure, KETOCONAZOLE_SMILES)
    perturbed = copy.deepcopy(mol)
    conf = perturbed.GetConformer()
    rng = np.random.default_rng(0)
    for i in range(perturbed.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        noise = rng.normal(scale=1.5, size=3)
        conf.SetAtomPosition(i, (pos.x + noise[0], pos.y + noise[1], pos.z + noise[2]))

    rmsd = ligand_pose_rmsd(copy.deepcopy(mol), perturbed)
    assert rmsd > 1.0  # real per-atom noise must not be hidden by alignment


def test_ligand_pose_rmsd_between_predictions_matches_direct_call():
    from cypmode.validation.pose_comparison import ligand_pose_rmsd_between_predictions

    structure = load_structure(CIF_PATH)
    rmsd = ligand_pose_rmsd_between_predictions(structure, load_structure(CIF_PATH), KETOCONAZOLE_SMILES)
    assert rmsd == pytest.approx(0.0, abs=1e-6)  # same real structure loaded twice
