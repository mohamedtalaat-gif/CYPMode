# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("Bio")

from cypmode.validation.structures import load_structure

RESULT_DIR = Path("data/structure_track/outputs/out/cyp3a4/boltz_results_cyp3a4_ketoconazole")
CIF_PATH = RESULT_DIR / "predictions" / "cyp3a4_ketoconazole" / "cyp3a4_ketoconazole_model_0.cif"
KETOCONAZOLE_SMILES = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"

pytestmark = pytest.mark.skipif(
    not CIF_PATH.exists(),
    reason="requires a local Boltz-2 run; see README.md's Validation section",
)


def _rigid_transform(structure, rotation: np.ndarray, translation: np.ndarray):
    """Applies one real rigid transform to every atom (protein, heme, and
    ligand alike) -- simulates the same complex predicted in a different
    global orientation/position, not a different binding pose.
    """
    moved = copy.deepcopy(structure)
    for atom in moved.get_atoms():
        atom.transform(rotation, translation)
    return moved


def test_bisy_rmsd_is_zero_for_an_identical_structure_pair():
    from cypmode.validation.bisy_rmsd import bisy_rmsd

    structure = load_structure(CIF_PATH)
    rmsd = bisy_rmsd(structure, load_structure(CIF_PATH), KETOCONAZOLE_SMILES)
    assert rmsd == pytest.approx(0.0, abs=1e-3)


def test_bisy_rmsd_is_invariant_to_a_global_rigid_transform_of_the_whole_complex():
    """The real, discriminating test: rotating/translating the ENTIRE
    complex (protein + heme + ligand together) changes every atom's raw
    coordinates, but the ligand's position RELATIVE to the binding
    pocket has not changed at all. A correct pocket-superposed BiSyRMSD
    must still report ~0 here -- a naive RMSD on raw, un-superposed
    coordinates would not.
    """
    from cypmode.validation.bisy_rmsd import bisy_rmsd

    structure = load_structure(CIF_PATH)
    rng = np.random.default_rng(0)
    # a real rotation matrix via QR decomposition of a random matrix
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    translation = rng.normal(scale=20.0, size=3)

    moved = _rigid_transform(structure, rotation, translation)
    rmsd = bisy_rmsd(structure, moved, KETOCONAZOLE_SMILES)
    assert rmsd == pytest.approx(0.0, abs=1e-2)


def test_bisy_rmsd_detects_a_real_ligand_only_pose_difference():
    """Rigidly mistranslating only the ligand (the binding-site protein
    residues are untouched) must be detected as a real nonzero BiSyRMSD
    -- a real, physically meaningful failure mode (a docking pose
    shifted out of the true pocket), and one that preserves the
    ligand's own bond lengths exactly, unlike per-atom random noise,
    which can perturb bond lengths enough that RDKit's proximity-based
    bond perception (inside _ligand_mol_3d) no longer recovers the
    correct connectivity -- a re-perception artifact, not something
    bisy_rmsd itself should be judged against.
    """
    from cypmode.validation.bisy_rmsd import bisy_rmsd
    from cypmode.validation.structures import _ligand_atoms

    structure = load_structure(CIF_PATH)
    perturbed = copy.deepcopy(structure)
    shift = np.array([3.0, 0.0, 0.0])
    for atom in _ligand_atoms(perturbed):
        atom.set_coord(atom.get_coord() + shift)

    rmsd = bisy_rmsd(structure, perturbed, KETOCONAZOLE_SMILES)
    assert rmsd == pytest.approx(3.0, abs=0.1)  # a pure rigid shift's RMSD equals the shift magnitude


def test_binding_site_ca_atoms_finds_real_pocket_residues():
    from cypmode.validation.bisy_rmsd import binding_site_ca_atoms

    structure = load_structure(CIF_PATH)
    pocket = binding_site_ca_atoms(structure)
    assert len(pocket) >= 3  # a real ~450-residue CYP3A4 pocket has several contacting residues
    assert all(atom.get_name() == "CA" for atom in pocket.values())


def test_bisy_rmsd_raises_when_too_few_binding_site_residues_match():
    from cypmode.validation.bisy_rmsd import bisy_rmsd

    structure = load_structure(CIF_PATH)
    with pytest.raises(ValueError, match="binding-site residues"):
        bisy_rmsd(structure, load_structure(CIF_PATH), KETOCONAZOLE_SMILES, ligand_cutoff=0.01)
