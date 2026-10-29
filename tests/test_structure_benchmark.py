# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

from pathlib import Path

import pandas as pd
import pytest

from cypmode.validation.structure_benchmark import (
    MAX_HEAVY,
    MIN_HEAVY,
    candidates,
    panel_seed_smiles,
    pick_diverse,
)

REFS = [
    {"comp_id": "AAA", "smiles": "c1ccccc1CCN1CCCC1C(=O)O", "n_heavy": 22, "resolution": 2.5, "pdb_id": "1AAA"},
    {"comp_id": "AAA", "smiles": "c1ccccc1CCN1CCCC1C(=O)O", "n_heavy": 22, "resolution": 2.0, "pdb_id": "1AAB"},  # better copy wins
    {"comp_id": "BBB", "smiles": "CC(C)(C)OC(=O)NCCCCCCNC(=O)c1cccnc1", "n_heavy": 25, "resolution": 2.1, "pdb_id": "1BBB"},
    {"comp_id": "CCC", "smiles": "C1CCC2C(C1)CCC1C2CCC2CCCC21", "n_heavy": 20, "resolution": 2.2, "pdb_id": "1CCC"},
    {"comp_id": "TINY", "smiles": "CCO", "n_heavy": 3, "resolution": 1.5, "pdb_id": "1TIN"},  # too small
    {"comp_id": "METAL", "smiles": "[Fe]C1=CC=CC=C1CCCCCCCCCC", "n_heavy": 21, "resolution": 2.0, "pdb_id": "1MET"},  # organometallic
    {"comp_id": "LOWRES", "smiles": "c1ccc2ccccc2c1CCCCCCCCC", "n_heavy": 21, "resolution": 3.9, "pdb_id": "1LOW"},  # poorly resolved
    {"comp_id": "SALT", "smiles": "CCCCCCCCCCCCCCCCCCCC.Cl", "n_heavy": 21, "resolution": 2.0, "pdb_id": "1SAL"},  # two fragments
    {"comp_id": "NOSMILES", "smiles": None, "n_heavy": 30, "resolution": 2.0, "pdb_id": "1NOS"},
]


def test_candidates_apply_the_drug_likeness_and_quality_criteria_and_keep_the_best_copy():
    table = candidates(REFS)
    assert set(table["comp_id"]) == {"AAA", "BBB", "CCC"}
    assert table.set_index("comp_id").loc["AAA", "pdb_id"] == "1AAB"
    assert table["n_heavy"].between(MIN_HEAVY, MAX_HEAVY).all()


def test_pick_diverse_is_deterministic_and_never_returns_a_seed_or_duplicates():
    table = candidates(REFS)
    seeds = ["c1ccccc1CCN1CCCC1C(=O)O"]  # identical to AAA: AAA is no longer far from the already-predicted set
    first = pick_diverse(table, seeds, 2)
    again = pick_diverse(table, seeds, 2)
    assert first["comp_id"].tolist() == again["comp_id"].tolist()
    assert len(set(first["comp_id"])) == 2
    assert isinstance(first, pd.DataFrame) and "AAA" not in first["comp_id"].tolist()[:1]


REFERENCE_SET = Path("data/structure_track/inputs/cyp3a4_reference_ligands.json")


@pytest.mark.skipif(not REFERENCE_SET.exists(), reason="the reference set of the PDB is not on this machine")
def test_the_committed_benchmark_is_what_the_picker_returns_from_the_panel_seeds():
    from cypmode.validation.reference_ligands import load_reference_set

    picked = pick_diverse(candidates(load_reference_set()), panel_seed_smiles(), 10)
    committed = pd.read_csv("data/structure_track/inputs/structure_benchmark.csv")
    assert [f"bench-{c}" for c in picked["comp_id"]] == committed["Molecule_Name"].tolist()  # the same ten, in the same order
    assert picked["smiles"].tolist() == committed["SMILES"].tolist()
