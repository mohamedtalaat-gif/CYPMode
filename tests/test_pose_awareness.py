import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cypmode.validation import pose_awareness as pa


def test_window_penalty_is_zero_inside_the_substrate_window_and_linear_outside():
    assert pa.window_penalty(4.5) == 0.0 and pa.window_penalty(3.5) == 0.0 and pa.window_penalty(6.0) == 0.0
    assert pa.window_penalty(2.0) == pytest.approx(1.5) and pa.window_penalty(8.0) == pytest.approx(2.0)


def test_contacts_count_packing_and_clashes_distinguish_hydrogen_bonds_from_overlaps():
    protein = np.array([[0.0, 0.0, 3.8], [0.0, 0.0, 12.0]])
    ligand = np.array([[0.0, 0.0, 0.0]])
    out = pa.contact_terms(ligand, ["C"], protein, ["C", "C"])
    assert out["contacts"] == 1.0 and out["clashes"] == 0.0  # one protein atom within 4.5 A, none closer than 3.0 A
    overlap = pa.contact_terms(ligand, ["C"], np.array([[0.0, 0.0, 2.5]]), ["C"])
    assert overlap["clashes"] == 1.0
    hbond = pa.contact_terms(ligand, ["O"], np.array([[0.0, 0.0, 2.8]]), ["N"])
    assert hbond["clashes"] == 0.0  # a polar pair at 2.8 A is a hydrogen bond, not a clash
    assert pa.contact_terms(ligand, ["O"], np.array([[0.0, 0.0, 2.2]]), ["N"])["clashes"] == 1.0


def test_pharmacophore_terms_use_the_donor_for_type_ii_ligands_and_the_site_otherwise():
    mol = Chem.AddHs(Chem.MolFromSmiles("Cn1ccnc1"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    pose = Chem.RemoveHs(mol)
    xyz = pose.GetConformer().GetPositions()
    donor = [a.GetIdx() for a in pose.GetAtoms() if a.GetSymbol() == "N" and a.GetDegree() == 2][0]
    iron = xyz[donor] + np.array([0.0, 0.0, 2.2])
    out = pa.pharmacophore_terms(pose, "Cn1ccnc1", iron, (donor, "N1"), None)
    assert out["donor_error"] == pytest.approx(0.0, abs=1e-6) and np.isnan(out["site_error"])
    far = pa.pharmacophore_terms(pose, "Cn1ccnc1", xyz[0] + np.array([0.0, 0.0, 9.0]), None, 0)
    assert np.isnan(far["donor_error"]) and far["site_error"] == pytest.approx(3.0, abs=1e-6)


def test_within_pool_scores_rank_better_poses_higher_and_combine_without_fitted_weights():
    pool = pd.DataFrame({
        "compound": "a", "true_rmsd": [1.0, 3.0, 5.0], "mbr": [0.9, 0.5, 0.2],
        "conf_rmsd": [0.3, 1.0, 2.0], "conf_energy": [0.5, 3.0, 6.0], "burial": [0.9, 0.7, 0.4], "contacts": [3.0, 2.0, 1.0], "clashes": [0.0, 1.0, 3.0],
        "donor_error": [0.1, 0.8, 1.5], "site_error": np.nan,
    })
    s = pa.within_pool_scores(pool)
    assert s.physics.is_monotonic_decreasing and s.pharmacophore.is_monotonic_decreasing and s.combined.is_monotonic_decreasing


def test_evaluate_reports_a_positive_correlation_for_informative_terms_and_gate_needs_three_ligands():
    rng = np.random.default_rng(0)
    rows = []
    for ligand in ("a", "b", "c", "d"):
        rmsd = rng.uniform(1, 6, 6)
        for r in rmsd:
            rows.append({"compound": ligand, "true_rmsd": r, "mbr": rng.random(), "conf_rmsd": r * 0.3 + rng.normal(0, 0.05), "conf_energy": r + rng.normal(0, 0.2),
                         "burial": 1 - r / 8, "contacts": 3 - r / 4, "clashes": r / 3, "donor_error": r / 4, "site_error": np.nan})
    _, per = pa.evaluate(pd.DataFrame(rows))
    assert (per.rho_physics > 0.8).all() and pa.gate(per, "rho_physics")["met"] is True
    assert pa.gate(per.head(2), "rho_physics")["met"] is False


def test_base_ligand_strips_run_suffixes():
    assert pa.base_ligand("bench-PK9-seed2") == "bench-PK9" and pa.base_ligand("bench-MWV-fecontact") == "bench-MWV" and pa.base_ligand("bench-TCI") == "bench-TCI"
