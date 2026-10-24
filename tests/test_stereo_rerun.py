# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import json

import pandas as pd

from cypmode.validation import stereo_rerun as sr

OFFICIAL = pd.DataFrame({"Molecule_Name": ["OCNT-A", "OCNT-B"], "SMILES": ["CC[C@@H](C)N", "CCOC"]})


def write_input(folder, run, smiles):
    (folder / f"cyp3a4_{run}.yaml").write_text(f"sequences:\n  - protein:\n      id: A\n      sequence: MKT\n  - ligand:\n      id: B\n      smiles: '{smiles}'\n")


def build(tmp_path):
    inputs, boltz, of3, final = tmp_path / "inputs", tmp_path / "out" / "cyp3a4", tmp_path / "openfold3", tmp_path / "final" / "runs"
    for folder in (inputs, boltz, of3 / "OCNT-A", of3 / "OCNT-B", final):
        folder.mkdir(parents=True)
    for run, smiles in (("OCNT-A", "CCC(C)N"), ("OCNT-A-u", "CCC(C)N"), ("OCNT-A-rs", "CC[C@H](C)N"), ("OCNT-B", "CCOC")):
        write_input(inputs, run, smiles)
        (boltz / f"boltz_results_cyp3a4_{run}").mkdir()
        (boltz / f"boltz_results_cyp3a4_{run}" / "pose.cif").write_text(run)
    base = tmp_path / "structure_track_results.json"
    base.write_text(json.dumps({"OCNT-A": {"x": 1}, "OCNT-B": {"x": 2}}))
    (final / "base_unguided_results.json").write_text(json.dumps({"OCNT-A-u": {"x": 3}, "OCNT-B-u": {"x": 4}}))
    return {"inputs": inputs, "boltz": boltz, "openfold3": of3, "base_results": base, "run_results": final, "target": tmp_path / "set_aside"}


def test_a_run_is_stale_when_its_input_differs_from_the_official_smiles_including_the_enantiomer(tmp_path):
    paths = build(tmp_path)
    assert sr.stale_runs(OFFICIAL, paths["inputs"]) == {"OCNT-A": ["OCNT-A", "OCNT-A-u", "OCNT-A-rs"]}  # the run of the other enantiomer is stale too


def test_the_stale_runs_are_set_aside_unchanged_and_leave_the_checkpoints(tmp_path):
    paths = build(tmp_path)
    stale = sr.stale_runs(OFFICIAL, paths["inputs"])
    kwargs = {k: v for k, v in paths.items() if k != "inputs"}
    ledger = sr.supersede(stale, **kwargs)
    assert len(ledger["moved"]) == 4  # three Boltz runs and the OpenFold3 folder of the ligand
    assert not (paths["boltz"] / "boltz_results_cyp3a4_OCNT-A").exists() and (paths["boltz"] / "boltz_results_cyp3a4_OCNT-B").exists()
    assert (paths["target"] / "boltz" / "boltz_results_cyp3a4_OCNT-A-u" / "pose.cif").read_text() == "OCNT-A-u"
    assert (paths["target"] / "openfold3" / "OCNT-A").exists() and (paths["openfold3"] / "OCNT-B").exists()
    assert json.loads(paths["base_results"].read_text()) == {"OCNT-B": {"x": 2}}
    assert json.loads((paths["run_results"] / "base_unguided_results.json").read_text()) == {"OCNT-B-u": {"x": 4}}
    assert json.loads((paths["target"] / "checkpoints" / "structure_track_results.json").read_text()) == {"OCNT-A": {"x": 1}, "OCNT-B": {"x": 2}}  # the original, before the entry left
    assert json.loads((paths["target"] / "ledger.json").read_text())[0]["ligands"] == {"OCNT-A": ["OCNT-A", "OCNT-A-u", "OCNT-A-rs"]}


def test_setting_aside_twice_changes_nothing_the_second_time(tmp_path):
    paths = build(tmp_path)
    stale = sr.stale_runs(OFFICIAL, paths["inputs"])
    kwargs = {k: v for k, v in paths.items() if k != "inputs"}
    sr.supersede(stale, **kwargs)
    second = sr.supersede(stale, **kwargs)
    assert second["moved"] == [] and second["checkpoints"] == []
    assert len(json.loads((paths["target"] / "ledger.json").read_text())) == 1


def test_a_dry_run_only_lists(tmp_path):
    paths = build(tmp_path)
    ledger = sr.supersede(sr.stale_runs(OFFICIAL, paths["inputs"]), dry_run=True, **{k: v for k, v in paths.items() if k != "inputs"})
    assert len(ledger["moved"]) == 4 and (paths["boltz"] / "boltz_results_cyp3a4_OCNT-A").exists()
    assert json.loads(paths["base_results"].read_text()).keys() == {"OCNT-A", "OCNT-B"} and not paths["target"].exists()


def test_nothing_is_stale_once_the_runs_were_made_from_the_official_smiles(tmp_path):
    paths = build(tmp_path)
    for run in ("OCNT-A", "OCNT-A-u", "OCNT-A-rs"):
        write_input(paths["inputs"], run, "CC[C@@H](C)N")
    assert sr.stale_runs(OFFICIAL, paths["inputs"]) == {}


def test_the_pick_report_tells_the_unchanged_from_the_changed_poses(tmp_path, monkeypatch):
    boltz = tmp_path / "out" / "cyp3a4"
    target = tmp_path / "set_aside"
    (boltz / "run_a").mkdir(parents=True)
    (boltz / "run_b").mkdir(parents=True)
    (target / "boltz" / "run_b").mkdir(parents=True)
    (boltz / "run_a" / "m.cif").write_text("same")
    (target / "boltz" / "run_b" / "m.cif").write_text("old")  # the earlier pick of B was set aside
    (boltz / "run_b" / "m.cif").write_text("new")
    monkeypatch.setattr("cypmode.validation.structures.load_structure", lambda path: path.read_text())
    monkeypatch.setattr("cypmode.validation.pose_consensus.aligned_ligand_rmsd", lambda a, b, smiles: {"ligand_rmsd": 3.0 if (a, b) == ("old", "new") else -1.0})
    monkeypatch.setattr("cypmode.validation.pose_consensus.pose_metrics", lambda structure, smiles: {"closest_motif_n_fe": {"old": 2.1, "new": 2.3}[structure]})
    before = pd.DataFrame({"compound": ["A", "B"], "chosen_cif": [str(boltz / "run_a" / "m.cif"), str(boltz / "run_b" / "m.cif")]})
    after = pd.DataFrame({"compound": ["A", "B"], "chosen_cif": [str(boltz / "run_a" / "m.cif"), str(boltz / "run_b" / "m.cif")]})
    report = sr.pick_report(before, after, {"A": "C", "B": "CC"}, boltz=boltz, target=target)
    assert report.changed.tolist() == [False, True]
    assert report.loc[1, "ligand_rmsd"] == 3.0 and (report.loc[1, "fe_n_old"], report.loc[1, "fe_n_new"]) == (2.1, 2.3)
