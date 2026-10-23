# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")

from cypmode.validation import conformer_bias as bias

SMILES = "CCCCCc1ccccc1"  # pentylbenzene: a flexible chain on a ring


def _record(conformer_rank: str):
    """A crystal record whose pose is a conformer of the molecule itself: the lowest-energy one or the highest-energy one of 60."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(SMILES))
    ids = list(AllChem.EmbedMultipleConfs(mol, 60, randomSeed=7))
    energy = np.array([e for _, e in AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)])
    pick = int(np.argmin(energy) if conformer_rank == "lowest" else np.argmax(energy))
    heavy = Chem.RemoveHs(Chem.Mol(mol, False, ids[pick]))
    xyz = heavy.GetConformer().GetPositions()
    return {"pdb_id": "TEST", "comp_id": "PNT", "n_heavy": heavy.GetNumAtoms(), "smiles": SMILES, "elements": [a.GetSymbol() for a in heavy.GetAtoms()], "coords": xyz.tolist()}, float(energy[pick] - energy.min())


def test_the_lowest_energy_conformer_as_crystal_pose_is_found_at_zero_energy():
    record, _ = _record("lowest")
    row = bias.copy_row(record)
    assert row["nearest_rmsd"] < 0.3 and row["needed_energy"] < 1.0 and row["rotatable"] == 4


def test_a_strained_crystal_pose_needs_a_higher_energy_window_than_the_lowest_conformer():
    low, _ = _record("lowest")
    high, strain = _record("highest")
    if strain < 0.5:
        pytest.skip("the 60 conformers of this molecule do not differ in energy")
    assert bias.copy_row(high)["needed_energy"] >= bias.copy_row(low)["needed_energy"]


def test_a_record_that_cannot_be_built_gives_nan_and_is_counted_as_not_evaluated():
    row = bias.copy_row({"pdb_id": "X", "comp_id": "BAD", "n_heavy": 3, "smiles": "not a smiles", "elements": ["C"], "coords": [[0, 0, 0]]})
    assert np.isnan(row["nearest_rmsd"])


def test_the_summary_applies_the_consequence_rule_to_the_coverage_of_the_ensemble_of_the_block():
    base = {"pdb_id": "A", "n_heavy": 20, "rotatable": 3, "nearest_energy": 2.0}
    covered = pd.DataFrame([{**base, "comp_id": f"C{i}", "nearest_rmsd": 0.5, "needed_energy": 3.0 + i, "ensemble_rmsd": 0.8} for i in range(10)])
    missed = covered.copy()
    missed.loc[:5, "ensemble_rmsd"] = 2.5
    good = "\n".join(bias.summarize(covered))
    bad = "\n".join(bias.summarize(missed))
    assert "block_ensemble_covers_the_bound_state (at least 80% of the copies within 1.0 A): YES" in good and "consequence" not in good
    assert ": NO" in bad and "misses the bound conformation of 60% of the crystal ligands" in bad
    assert "within 5 kcal/mol of the lowest conformer for 3 of 10 covered copies" in good


def test_the_pose_strain_lines_report_the_share_of_poses_that_resemble_no_low_energy_conformer(tmp_path):
    table = pd.DataFrame({"conf_rmsd": [0.4, 0.8, 1.5, 2.5], "conf_energy": [1.0, 2.0, 3.0, 4.0]})
    path = tmp_path / "poses.csv"
    table.to_csv(path, index=False)
    line = bias.pose_strain_lines(path)[0]
    assert "farther than 1.0 A for 50%" in line and "farther than 2.0 A for 25%" in line
    assert bias.pose_strain_lines(tmp_path / "missing.csv") == []
