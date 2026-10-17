import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cypmode.validation import final_recipe
from cypmode.validation.restraint_expansion import (
    PLAN_PATH,
    eligible,
    load_plan,
    paired_statistics,
    per_ligand,
    series_of,
)

PYRIDINE_AMIDE = "CC(=O)NCc1cccnc1"  # a bare aromatic nitrogen: a Type II donor, 10 heavy atoms
NO_DONOR = "CC(=O)NCc1ccccc1"
BIG_DONOR = "CC(C)(C)OC(=O)NCCCCCC(=O)NCc1cccnc1"  # 23 heavy atoms
TOO_BIG = "c1ccc(cc1)" + "C(=O)Nc2ccccc2" * 5 + "c3cccnc3"


def record(comp, smiles, resolution=2.0, name="x"):
    return {"comp_id": comp, "name": name, "smiles": smiles, "resolution": resolution, "pdb_id": "1W0F", "min_fe_distance": 2.2}


def test_the_plan_is_registered_with_the_decision_rules_the_code_reads():
    plan = load_plan()
    assert PLAN_PATH.exists() and plan["registered"] == "2026-10-12"
    assert plan["endpoints"]["alpha"] == 0.05 and plan["design"]["samples_per_condition"] == 2
    assert set(plan["eligibility"]["strata"]) == {"primary", "secondary"}


def test_only_ligands_with_a_donor_inside_the_size_and_resolution_limits_are_eligible():
    refs = [record("DON", PYRIDINE_AMIDE), record("NOD", NO_DONOR), record("BIG", TOO_BIG), record("LOW", BIG_DONOR, resolution=3.9), record("OK2", BIG_DONOR, resolution=2.5)]
    table = eligible(refs, benchmark_smiles=[])
    assert list(table["comp_id"]) == ["DON", "OK2"]
    assert set(table["stratum"]) == {"primary"}


def test_a_ligand_of_the_benchmark_is_not_eligible():
    table = eligible([record("DON", PYRIDINE_AMIDE)], benchmark_smiles=[PYRIDINE_AMIDE])
    assert table.empty


def test_organometallic_and_multi_fragment_ligands_are_not_eligible():
    refs = [record("MET", "C[Ir]"), record("TWO", PYRIDINE_AMIDE + ".O")]
    assert eligible(refs, benchmark_smiles=[]).empty


def test_one_row_per_distinct_ligand_with_the_best_resolution():
    refs = [record("DON", PYRIDINE_AMIDE, resolution=3.0), record("DON", PYRIDINE_AMIDE, resolution=2.0)]
    table = eligible(refs, benchmark_smiles=[])
    assert len(table) == 1 and table["resolution"].iloc[0] == 2.0


def test_the_test_confirms_a_consistent_gain_and_detects_a_consistent_loss():
    better = paired_statistics(np.array([0.1, 0.2, 0.15, 0.3, 0.12, 0.25, 0.18, 0.22]), alpha=0.05)
    worse = paired_statistics(-np.array([0.1, 0.2, 0.15, 0.3, 0.12, 0.25, 0.18, 0.22]), alpha=0.05)
    assert better["confirmed"] and not better["harm"]
    assert worse["harm"] and not worse["confirmed"]


def test_one_big_win_among_losses_is_not_a_confirmation():
    stats = paired_statistics(np.array([1.0, -0.05, -0.04, -0.06, 0.01, -0.02]), alpha=0.05)
    assert not stats["confirmed"] and not stats["harm"]


def test_the_score_of_a_ligand_is_the_mean_over_its_samples_in_each_condition():
    poses = pd.DataFrame({"compound": ["exp-A"] * 3 + ["exp-A-febond"] * 3, "engine": ["boltz"] * 6, "sample": [0, 1, 2, 0, 1, 2], "lddt_pli": [0.2, 0.4, 0.9, 0.6, 0.8, 0.1]})
    wide = per_ligand(poses, "lddt_pli", samples=2)
    assert wide.loc["exp-A", "unguided"] == pytest.approx(0.3) and wide.loc["exp-A", "restrained"] == pytest.approx(0.7)
    assert wide.loc["exp-A", "difference"] == pytest.approx(0.4)


def test_a_stratum_with_no_scored_poses_gives_an_empty_table_instead_of_failing():
    none = pd.DataFrame({"compound": pd.Series([], dtype=str), "engine": pd.Series([], dtype=str), "sample": pd.Series([], dtype=int), "lddt_pli": pd.Series([], dtype=float)})
    wide = per_ligand(none, "lddt_pli", samples=2)
    assert len(wide) == 0 and list(wide.columns) == ["unguided", "restrained", "difference"]
    assert paired_statistics(wide["difference"].to_numpy(), alpha=0.05)["confirmed"] is False


def test_close_analogs_form_one_series():
    ligands = pd.DataFrame({"comp_id": ["A", "B", "C"], "smiles": [PYRIDINE_AMIDE, "CC(=O)NCc1cccnc1C", "O=C(c1ccccc1)Nc1ccc(Br)cc1Cl"]})
    series = series_of(ligands, 0.3)
    assert series["exp-A"] == series["exp-B"] != series["exp-C"]


def test_the_recipe_withdraws_the_bond_restraint_when_harm_is_found_on_the_new_ligands():
    assert final_recipe.decide({"restraint_bond": True}).restraint == "bond"
    withdrawn = final_recipe.decide({"restraint_bond": True, "restraint_bond_harm": True})
    assert withdrawn.restraint is None and any("withdrawn" in reason for reason in withdrawn.reasons)
    assert final_recipe.decide({"restraint_bond": True, "restraint_bond_harm": False}).restraint == "bond"


def test_the_plan_file_is_valid_json_with_a_series_threshold():
    assert 0 < json.loads(Path(PLAN_PATH).read_text())["endpoints"]["series_similarity_threshold"] < 1
