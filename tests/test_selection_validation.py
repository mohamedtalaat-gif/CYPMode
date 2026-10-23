# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd
import pytest

from cypmode.validation import pose_pools as pp
from cypmode.validation import selection_validation as sv


def test_mbr_picks_the_consensus_pose_and_disagreement_falls_as_the_pool_agrees():
    # poses 0-2 agree with each other (0.9), pose 3 is an outlier (0.2 with everyone)
    m = np.full((4, 4), 0.2)
    m[:3, :3] = 0.9
    np.fill_diagonal(m, 1.0)
    assert pp.mbr_pick(m) in (0, 1, 2) and pp.mbr_pick(m, [2, 3]) in (2, 3)
    assert pp.pool_disagreement(m, [0, 1, 2]) == pytest.approx(0.1) and pp.pool_disagreement(m, [0]) == 1.0
    assert pp.pool_disagreement(m, [0, 1, 2, 3]) > pp.pool_disagreement(m, [0, 1, 2])


def test_paired_gate_requires_consistent_improvement_not_one_big_win():
    base = pd.Series([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    consistent = pd.Series([3.0, 3.2, 3.1, 3.4, 2.9, 3.3, 3.0, 3.2])
    one_big = pd.Series([0.5, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1])
    assert sv.paired_gate(base, consistent)["met"] is True
    g = sv.paired_gate(base, one_big)
    assert g["mean_improvement"] > 0 and g["min_drop_one"] < 0 and g["met"] is False  # driven by a single ligand
    assert sv.paired_gate(base, base + 0.5)["met"] is False


def test_evaluate_rules_scores_the_picked_pose_by_its_true_rmsd():
    table = pd.DataFrame({"compound": ["a"] * 4, "engine": ["boltz"] * 4, "sample": [0, 1, 2, 3], "true_rmsd": [6.0, 1.0, 1.2, 5.0]})
    m = np.full((4, 4), 0.3)
    m[1:3, 1:3] = 0.9  # samples 1 and 2 agree and are the good poses
    np.fill_diagonal(m, 1.0)
    out = sv.evaluate_rules(table, {"a": ([("boltz", i) for i in range(4)], m)})
    assert out.loc[0, "first_boltz"] == 6.0 and out.loc[0, "mbr_boltz"] in (1.0, 1.2) and out.loc[0, "oracle_boltz"] == 1.0
    assert out.loc[0, "mean_pool_boltz"] == pytest.approx(3.3)


def test_engine_gate_needs_different_errors_and_a_better_pick():
    rng = np.random.default_rng(0)
    n = 10
    first = rng.uniform(1, 6, n)
    outcomes = pd.DataFrame({"first_boltz": first, "mbr_boltz": first, "first_openfold3": rng.uniform(1, 6, n), "mbr_both": first - 0.5 - rng.uniform(0, 0.3, n)})
    g = sv.gates(outcomes)["engine_consensus"]
    assert g["met"] == (abs(g["errors_correlation"]) < 0.7 and g["mean_improvement"] > 0) or g["met"] is False
    correlated = outcomes.assign(first_openfold3=first + 0.01 * rng.normal(size=n))
    assert sv.gates(correlated)["engine_consensus"]["met"] is False  # identical errors: a second engine adds nothing


def _two_engine_case(added: str):
    """Boltz samples 0-1 are far from the crystal pose, the added engine's samples 2-3 are close and agree with each other."""
    table = pd.DataFrame({"compound": ["a"] * 4, "engine": ["boltz", "boltz", added, added], "sample": [0, 1, 0, 1], "true_rmsd": [6.0, 5.0, 1.0, 1.5]})
    m = np.full((4, 4), 0.2)
    m[2:, 2:] = 0.9
    np.fill_diagonal(m, 1.0)
    return table, {"a": ([("boltz", 0), ("boltz", 1), (added, 0), (added, 1)], m)}


def test_the_added_engine_columns_are_named_after_the_engine():
    table, matrices = _two_engine_case("bioemu2")
    out = sv.evaluate_rules(table, matrices, engine="bioemu2")
    assert "first_bioemu2" in out and "first_openfold3" not in out
    assert out.loc[0, "first_bioemu2"] == 1.0 and out.loc[0, "mbr_both"] in (1.0, 1.5)
    default = sv.evaluate_rules(*_two_engine_case("openfold3"))  # the default engine keeps its old column
    assert "first_openfold3" in default


def test_the_gate_of_another_engine_has_its_own_name_and_keeps_the_openfold3_name_for_openfold3():
    outcomes = pd.DataFrame({"first_boltz": [4.0, 5.0, 6.0, 3.0], "mbr_boltz": [4.0, 5.0, 6.0, 3.0], "first_bioemu2": [1.0, 2.0, 1.0, 2.0],
                             "mbr_both": [3.0, 4.0, 5.0, 2.0]})
    assert "engine_consensus_bioemu2" in sv.gates(outcomes, "bioemu2") and "engine_consensus" not in sv.gates(outcomes, "bioemu2")
    renamed = outcomes.rename(columns={"first_bioemu2": "first_openfold3"})
    assert "engine_consensus" in sv.gates(renamed)


def test_pool_paths_lists_the_samples_of_the_bioemu2_engine(monkeypatch, tmp_path):
    from cypmode.validation import bioemu2

    monkeypatch.setattr(bioemu2, "sample_cifs", lambda name, target="cyp3a4": [tmp_path / "sample_0.cif", tmp_path / "sample_1.cif"])
    listed = pp.pool_paths("bench-X", ("bioemu2",))
    assert [(engine, index) for engine, index, _ in listed] == [("bioemu2", 0), ("bioemu2", 1)]
