# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import json
import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest

from cypmode.validation import relax_final

OK = {"passed": True, "margin": 0.85}


def test_the_rule_adopts_a_relaxed_pose_only_when_every_criterion_holds():
    info = {"rmsd": 0.5, "fe_n_before": 2.2, "fe_n_after": 2.3}
    assert relax_final.adopt(OK, {"passed": True, "margin": 0.90}, info) == (True, "adopted")
    assert relax_final.adopt({"passed": False, "margin": float("nan")}, {"passed": True, "margin": 0.9}, info)[0] is False
    assert relax_final.adopt(OK, {"passed": False, "margin": 0.9}, info)[1] == "the relaxed pose fails a check"
    assert relax_final.adopt(OK, {"passed": True, "margin": 0.85}, info)[1] == "the margin does not rise"  # equal is not a rise
    assert relax_final.adopt({"passed": True, "margin": 0.70}, {"passed": True, "margin": 0.78}, info)[1] == "the relaxed pose is below the margin floor"
    assert relax_final.adopt(OK, {"passed": True, "margin": 0.9}, {**info, "rmsd": 1.2})[1] == "the ligand moved too far"
    assert relax_final.adopt(OK, {"passed": True, "margin": 0.9}, {**info, "fe_n_after": 2.6})[1] == "the iron-nitrogen coordination changed"
    assert relax_final.adopt(OK, {"passed": True, "margin": 0.9}, {**info, "fe_n_after": float("nan")})[1] == "the iron-nitrogen coordination changed"
    assert relax_final.adopt(OK, {"passed": True, "margin": 0.9}, {"rmsd": 0.5, "fe_n_before": float("nan"), "fe_n_after": float("nan")})[0] is True  # no coordination to keep
    assert relax_final.adopt(OK, {"passed": True, "margin": 0.9}, {"error": "boom"})[1].startswith("the relaxation failed")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    final = tmp_path / "final"
    (final / "_pdb_files").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "gate.log").write_text("STRUCTURE GATE force_field_relaxation (validity, margin and fidelity at p < 0.05): MET\n")
    official = tmp_path / "official.csv"
    pd.DataFrame({"Molecule_Name": ["a", "b", "c"], "SMILES": ["C", "CC", "CCC"]}).to_csv(official, index=False)
    for n in "abc":
        (final / "_pdb_files" / f"{n}.pdb").write_text(f"ORIGINAL {n}\n")
    from cypmode.challenge.structure_submission import write_deterministic_zip

    write_deterministic_zip([final / "_pdb_files" / f"{n}.pdb" for n in "abc"], final / "structures_final.zip")
    (final / "manifest.json").write_text(json.dumps({"sha256": "old"}))
    monkeypatch.setattr(relax_final, "FINAL", final)
    monkeypatch.setattr(relax_final, "WORK", final / "relaxation")
    monkeypatch.setattr(relax_final, "OFFICIAL", official)
    monkeypatch.setattr(relax_final, "GATE_LOG", tmp_path / "logs" / "gate.log")
    movement = {"a": 0.4, "b": 0.5, "c": 1.5}  # c moves too far

    def relax_all(jobs):
        for job in jobs:
            out = job["pdb"].split("/")[-1][:-4]
            from pathlib import Path

            Path(job["out"]).write_text(f"RELAXED {out}\n")
            Path(job["out"] + ".json").write_text(json.dumps({"rmsd": movement[out], "fe_n_before": 2.2, "fe_n_after": 2.25}))

    def check_pdb(path, smiles):
        with open(path) as handle:
            text = handle.read()
        margin = 0.90 if text.startswith("RELAXED") else 0.85
        return SimpleNamespace(passed=True, distance_margin=margin, failed=[], error="")

    def check_zip(path, smiles_of):
        with zipfile.ZipFile(path) as z:
            return {name[:-4]: SimpleNamespace(passed=True, distance_margin=0.90 if z.read(name).startswith(b"RELAXED") else 0.85) for name in z.namelist()}

    def zip_reproduces_stereo(path, smiles_of):
        with zipfile.ZipFile(path) as z:
            return {name[:-4]: True for name in z.namelist()}

    monkeypatch.setattr("cypmode.validation.pose_stereo.zip_reproduces_stereo", zip_reproduces_stereo)
    monkeypatch.setattr("cypmode.validation.relax_gate.relax_all", relax_all)
    monkeypatch.setattr("cypmode.validation.official_posebusters.check_pdb", check_pdb)
    monkeypatch.setattr("cypmode.validation.official_posebusters.check_zip", check_zip)
    monkeypatch.setattr("cypmode.challenge.submission.portal_validate_structure", lambda p: (True, "ok"))
    return final


def members(path):
    with zipfile.ZipFile(path) as z:
        return {name: z.read(name).decode().strip() for name in z.namelist()}


def test_the_final_zip_takes_the_adopted_poses_and_keeps_the_rest(workspace):
    assert relax_final.main() == 0
    assert members(workspace / "structures_final.zip") == {"a.pdb": "RELAXED a", "b.pdb": "RELAXED b", "c.pdb": "ORIGINAL c"}
    assert members(workspace / "structures_unrelaxed.zip") == {"a.pdb": "ORIGINAL a", "b.pdb": "ORIGINAL b", "c.pdb": "ORIGINAL c"}
    decisions = pd.read_csv(workspace / "relaxation" / "decisions.csv").set_index("compound")
    assert decisions.adopted.to_dict() == {"a": True, "b": True, "c": False} and decisions.loc["c", "reason"] == "the ligand moved too far"
    manifest = json.loads((workspace / "manifest.json").read_text())
    assert manifest["relaxation"]["adopted"] == 2 and manifest["sha256"] == manifest["relaxation"]["final_sha256"] != manifest["recipe_zip_sha256"]
    assert (workspace / "_pdb_files" / "a.pdb").read_text() == "ORIGINAL a\n"  # the recipe's poses are never modified


def test_a_second_run_gives_the_same_zip_and_a_changed_original_is_relaxed_again(workspace):
    relax_final.main()
    first = (workspace / "structures_final.zip").read_bytes()
    relax_final.main()
    assert (workspace / "structures_final.zip").read_bytes() == first
    (workspace / "_pdb_files" / "a.pdb").write_text("ORIGINAL a, another pose\n")  # the recipe chose another pose: the zip of the recipe changes too
    from cypmode.challenge.structure_submission import write_deterministic_zip

    write_deterministic_zip([workspace / "_pdb_files" / f"{n}.pdb" for n in "abc"], workspace / "structures_final.zip")
    relax_final.main()
    assert members(workspace / "structures_final.zip")["a.pdb"] == "RELAXED a"
    assert (workspace / "relaxation" / "relaxed" / "a.pdb.source").read_text() != ""


def test_an_unexpected_final_zip_is_never_overwritten(workspace):
    (workspace / "structures_final.zip").write_bytes(b"not one of ours")
    assert relax_final.main() == 1
    assert (workspace / "structures_final.zip").read_bytes() == b"not one of ours"


def test_nothing_is_used_when_the_gate_is_not_met(workspace, tmp_path):
    (tmp_path / "logs" / "gate.log").write_text("STRUCTURE GATE force_field_relaxation (validity, margin and fidelity at p < 0.05): NOT MET\n")
    before = (workspace / "structures_final.zip").read_bytes()
    assert relax_final.main() == 0
    assert (workspace / "structures_final.zip").read_bytes() == before and not (workspace / "relaxation").exists()


def test_a_candidate_zip_with_a_pose_of_the_other_enantiomer_replaces_nothing(workspace, monkeypatch):
    monkeypatch.setattr("cypmode.validation.pose_stereo.zip_reproduces_stereo", lambda path, smiles_of: {"a": True, "b": False, "c": None})
    before = (workspace / "structures_final.zip").read_bytes()
    assert relax_final.main() == 1
    assert (workspace / "structures_final.zip").read_bytes() == before
