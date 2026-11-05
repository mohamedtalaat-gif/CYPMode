# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from cypmode.post_deadline import structure
from cypmode.post_deadline.released import DEST


def _atom(record, serial, name, residue, chain, x, element):
    return f"{record:<6}{serial:>5} {name:<4} {residue:>3} {chain}{1:>4}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2}"


def test_the_gate_compares_the_mean_lddt_and_the_mean_rmsd_with_the_board_within_the_registered_tolerances():
    def summary(lddt, rmsd):
        return pd.DataFrame({"LDDT-PLI_mean": [lddt], "BiSyRMSD_mean": [rmsd], "LDDT-LP_mean": [0.8]}, index=pd.Index(["Structure"], name="Endpoint"))

    board = {"LDDT-PLI": 0.4133, "BiSyRMSD": 3.97, "LDDT-LP": 0.826}
    assert structure.gate(summary(0.4133, 3.97), board=board)["passed"]
    assert structure.gate(summary(0.4180, 3.90), board=board)["passed"]
    assert not structure.gate(summary(0.4300, 3.97), board=board)["passed"]
    assert not structure.gate(summary(0.4133, 4.20), board=board)["passed"]


def test_a_paired_difference_names_the_better_set_and_counts_the_compounds_in_which_it_is_higher():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.uniform(0.2, 0.6, 20))
    better = structure.paired_difference(base + 0.1, base, resamples=500)
    assert better["verdict"] == "first better" and better["first_higher"] == 20 and better["mean_difference"] == pytest.approx(0.1)
    same = structure.paired_difference(base, base, resamples=500)
    assert same["verdict"] == "no difference shown" and same["mean_difference"] == 0.0
    assert structure.paired_difference(base - 0.1, base, resamples=500)["verdict"] == "second better"


def test_the_metric_is_oriented_so_that_a_larger_value_is_better():
    values = pd.Series([1.0, 2.0])
    assert list(structure.oriented(values, "LDDT-PLI")) == [1.0, 2.0] and list(structure.oriented(values, "BiSyRMSD")) == [-1.0, -2.0]


def test_the_gaps_of_the_pool_are_taken_to_its_best_pose():
    pool = pd.DataFrame({"ligand": ["a", "a", "a", "b", "b"], "LDDT-PLI": [0.2, 0.6, 0.4, 0.1, 0.3]})
    table = structure.pool_gaps(pool, pd.Series({"a": 0.4, "b": 0.3}))
    assert table.loc["a", "best"] == 0.6 and table.loc["a", "random"] == pytest.approx(0.4) and table.loc["a", "poses"] == 3
    assert table.loc["a", "gap_submitted"] == pytest.approx(0.2) and table.loc["b", "gap_submitted"] == 0.0 and table.loc["b", "gap_random"] == pytest.approx(0.1)


def test_the_iron_nitrogen_distance_is_the_shortest_over_the_irons_and_the_ligand_nitrogens():
    text = "\n".join(
        [
            _atom("HETATM", 1, "FE", "HEM", "C", 0.0, "FE"),
            _atom("HETATM", 2, "FE", "HEM", "D", 50.0, "FE"),
            _atom("HETATM", 3, "N1", "LIG", "B", 2.1, "N"),
            _atom("HETATM", 4, "C1", "LIG", "B", 1.0, "C"),
            _atom("HETATM", 5, "N2", "LIG", "B", 9.0, "N"),
            _atom("ATOM", 6, "N", "ALA", "A", 0.5, "N"),
        ]
    )
    assert structure.iron_nitrogen_distance(text) == pytest.approx(2.1)
    assert structure.iron_nitrogen_distance(_atom("HETATM", 3, "N1", "LIG", "B", 2.1, "N")) is None  # no iron
    assert structure.iron_nitrogen_distance("") is None


def test_the_coordination_table_counts_the_calls_against_the_released_structures():
    truth = {"a": 2.1, "b": 2.2, "c": 6.0, "d": None}
    predicted = {"a": 2.0, "b": 5.5, "c": 2.4, "d": 7.0}
    table, counts = structure.coordination_table(truth, predicted)
    assert list(table["ligand"]) == ["a", "b", "c", "d"]
    assert counts.loc[True, True] == 1 and counts.loc[True, False] == 1 and counts.loc[False, True] == 1 and counts.loc[False, False] == 1


def test_the_resolution_is_read_from_the_header_of_a_released_structure():
    assert structure.resolution("REMARK   3 RESOLUTION: 3.50  \n") == 3.5 and structure.resolution("ATOM\n") is None


@pytest.mark.skipif(not (DEST / "organisers/evaluation/evaluate_predictions.py").exists(), reason="the evaluation code of the organisers is not in this checkout (python -m cypmode.post_deadline.released)")
def test_the_bootstrap_of_the_organisers_gives_the_mean_of_the_compounds_for_the_board_value():
    frame = pd.DataFrame({"LDDT-PLI": np.linspace(0.1, 0.7, 20), "BiSyRMSD": np.linspace(2.0, 8.0, 20), "LDDT-LP": np.linspace(0.6, 0.9, 20)})
    summary = structure.board_value(frame)
    assert summary.loc["Structure", "LDDT-PLI_mean"] == pytest.approx(frame["LDDT-PLI"].mean(), abs=0.01)
    assert summary.loc["Structure", "LDDT-PLI_std"] > 0


@pytest.mark.skipif(not (structure.OST_PYTHON.exists() and any(structure.REFERENCES.glob("*.pdb"))), reason="OpenStructure or the released structures are not on this machine")
def test_the_worker_scores_a_released_structure_against_itself_with_the_organisers_function(tmp_path):
    reference = min(structure.REFERENCES.glob("*.pdb"))
    smiles = structure._smiles()[reference.stem]
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"mode": "poses", "items": [{"key": "k", "model": str(reference), "reference": str(reference), "smiles": smiles}]}))
    done = subprocess.run([str(structure.OST_PYTHON), str(structure.WORKER), str(job), str(tmp_path / "out.json"), str(DEST / "organisers")], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr[-500:]
    row = json.loads((tmp_path / "out.json").read_text())["rows"][0]
    assert row["key"] == "k" and set(row) >= {"LDDT-PLI", "BiSyRMSD", "LDDT-LP", "pb_failed"}
