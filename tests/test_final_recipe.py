from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cypmode.validation import final_recipe as fr


def test_gate_lines_from_all_structure_logs_are_read(tmp_path):
    (tmp_path / "selection_validation.log").write_text("noise\nSTRUCTURE GATE mbr_selection (paired, drop-one robust, p < 0.1): MET\nSTRUCTURE GATE engine_consensus (x): NOT MET\n")
    (tmp_path / "structure_gates.log").write_text("STRUCTURE GATE restraint_bond: MET\nSTRUCTURE GATE pilot_seeds: NOT MET\n")
    gates = fr.read_gates(tmp_path)
    assert gates == {"mbr_selection": True, "engine_consensus": False, "restraint_bond": True, "pilot_seeds": False}


def test_recipe_follows_the_verdicts_and_defaults_to_the_old_behaviour_without_any():
    nothing = fr.decide({})
    assert (nothing.restraint, nothing.levers, nothing.openfold3, nothing.selection, nothing.ei_allocation) == (None, [], False, "first_valid", False)
    full = fr.decide({"restraint_bond": True, "restraint_contact": True, "pilot_seeds": True, "pilot_stepscale": True, "pilot_conformers": False,
                      "engine_consensus": True, "mbr_selection": True, "heme_frame_prior": True, "ei_sampling": True},
                     restraint_means={"restraint_bond": 3.5, "restraint_contact": 3.0})
    assert full.restraint == "contact" and full.levers == ["seeds", "stepscale"] and full.openfold3 and full.selection == "mbr_plus_prior" and full.ei_allocation
    assert fr.decide({"mbr_selection": True}).selection == "mbr"
    assert fr.decide({"heme_frame_prior": True}).selection == "first_valid"  # the prior only refines MBR, it never stands alone


def test_compute_estimate_counts_existing_samples_and_each_lever():
    existing = {"a": 1, "b": 5}
    base = fr.estimate_hours(fr.decide({}, extra_per_ligand=0), 2, existing)
    assert base == pytest.approx(4 * fr.HOURS_PER_BOLTZ_SAMPLE)  # only 'a' needs 4 more samples
    bigger = fr.estimate_hours(fr.decide({"pilot_seeds": True, "engine_consensus": True}, extra_per_ligand=0), 2, existing)
    assert bigger > base + 2 * fr.LEVER_RUNS * fr.HOURS_PER_BOLTZ_SAMPLE


def test_run_groups_only_contain_what_the_recipe_needs():
    official = pd.DataFrame({"Molecule_Name": ["a", "b"], "SMILES": ["CCO", "CCN"]})
    groups = fr.run_groups(fr.decide({}), official, {"a": 1, "b": 5}, [])
    assert [g["label"] for g in groups] == ["base_unguided"] and groups[0]["rows"].Molecule_Name.tolist() == ["a-u"] and groups[0]["rows"].Diffusion_Samples.tolist() == [4]
    groups = fr.run_groups(fr.decide({"restraint_bond": True}), official, {"a": 5, "b": 5}, [])
    assert [g["label"] for g in groups] == ["restraint_bond"] and groups[0]["flags"] == ["--donor-restraint", "bond"]
    assert groups[0]["rows"].Molecule_Name.tolist() == ["a-rs", "b-rs"]


def test_selection_prefers_the_valid_consensus_pose_and_falls_back_when_nothing_is_valid(monkeypatch):
    from cypmode.validation import pose_pools

    m = np.full((4, 4), 0.2)
    m[:3, :3] = 0.9
    np.fill_diagonal(m, 1.0)
    monkeypatch.setattr(pose_pools, "pairwise_lddt", lambda paths, smiles=None: m)
    pool = [("r", "boltz", Path(f"p{i}.cif")) for i in range(4)]
    valid_all = lambda p: True  # noqa: E731
    best, ok, _ = fr.select_pose(pool, "mbr", valid_all)
    assert best in (0, 1, 2) and ok
    best, ok, _ = fr.select_pose(pool, "mbr", lambda p: p.name != "p0.cif")  # the best consensus pose fails PoseBusters
    assert best in (1, 2) and ok
    best, ok, _ = fr.select_pose(pool, "mbr", lambda p: False)  # nothing valid: still returns a pose, flagged invalid
    assert best in range(4) and ok is False
    assert fr.select_pose(pool, "first_valid", lambda p: p.name == "p2.cif")[0] == 2


def test_pose_awareness_and_series_gates_add_rank_terms_to_the_selection():
    r = fr.decide({"pose_physics": True, "series_consensus": True})
    assert r.selection == "rank_terms" and r.terms == ["physics", "series"]
    r = fr.decide({"pose_combined": True, "pose_physics": False})
    assert r.terms == ["physics", "pharmacophore"]
    assert fr.decide({"mbr_selection": True}).terms == [] and fr.decide({}).selection == "first_valid"


def test_rank_terms_selection_uses_the_extra_scores_and_ignores_missing_ones(monkeypatch):
    from cypmode.validation import pose_pools

    m = np.full((4, 4), 0.5)
    np.fill_diagonal(m, 1.0)  # MBR cannot tell the poses apart
    monkeypatch.setattr(pose_pools, "pairwise_lddt", lambda paths, smiles=None: m)
    pool = [("r", "boltz", Path(f"p{i}.cif")) for i in range(4)]
    best, ok, _ = fr.select_pose(pool, "rank_terms", lambda p: True, None, {"physics": [0.1, 0.9, 0.2, 0.3]})
    assert best == 1 and ok
    best, _, _ = fr.select_pose(pool, "rank_terms", lambda p: True, None, {"series": [np.nan] * 4, "physics": [0.1, 0.2, 0.9, 0.3]})
    assert best == 2  # an all-missing term adds nothing, the available one decides


def test_ligand_table_links_official_ligands_to_donor_status_analogs_and_tdi_measurements():
    official = pd.DataFrame({"Molecule_Name": ["a", "b", "c"], "SMILES": ["Cn1ccnc1", "CCO", "c1ccccc1"]})
    train = pd.DataFrame({"SMILES": ["OCC", "CCCC"], "CYP3A4_pIC50_direct_inhibition": [5.5, 4.0], "CYP3A4_pIC50_TDI_condition": [6.1, 4.2]})
    test = pd.DataFrame({"SMILES": ["C1=CC=CC=C1"]})
    t = fr.ligand_table(official, train, test, [("a", "b", 0.6)])
    assert t.type_ii_donor.tolist() == [True, False, False] and t.analog.tolist()[:2] == ["b", "a"] and t.analog.isna().tolist() == [False, False, True]
    assert t.in_tdi_train.tolist() == [False, True, False] and t.in_tdi_test.tolist() == [False, False, True]
    assert t.cyp3a4_direct_pic50[1] == 5.5 and np.isnan(t.cyp3a4_direct_pic50[0])


def test_cryoem_lever_adds_template_runs_to_the_diversity_group(monkeypatch):
    rows = pd.DataFrame({"Molecule_Name": ["a-em1", "a-em2"], "SMILES": ["CCO", "CCO"], "Template_PDB": ["38FU", "38FV"], "Diffusion_Samples": [1, 1]})
    monkeypatch.setattr("cypmode.validation.cryoem_templates.load_records", lambda: {})
    monkeypatch.setattr("cypmode.validation.cryoem_templates.build_cryoem_table", lambda official, record, n_runs: rows)
    recipe = fr.decide({"pilot_cryoem": True})
    assert recipe.levers == ["cryoem"]
    official = pd.DataFrame({"Molecule_Name": ["a"], "SMILES": ["CCO"]})
    groups = fr.run_groups(recipe, official, {"a": 5}, [])
    assert [g["label"] for g in groups] == ["diversity_levers"] and groups[0]["rows"].Molecule_Name.tolist() == ["a-em1", "a-em2"]


def test_a_donor_ligand_reads_its_restrained_run_first_when_the_restraint_is_on():
    runs = ["L", "L-u", "L-rs", "L-ex1"]
    assert fr.run_order("L", runs, donor=True, restraint="bond") == ["L-rs", "L", "L-u", "L-ex1"]  # the restrained run first, the rest in order behind it
    assert fr.run_order("L", runs, donor=False, restraint="bond") == runs  # no donor, nothing to restrain
    assert fr.run_order("L", runs, donor=True, restraint=None) == runs  # the gate was not met
    assert fr.run_order("L", ["L", "L-u"], donor=True, restraint="bond") == ["L", "L-u"]  # no restrained run was produced
